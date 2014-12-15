// Copyright 2014 Facebook

#include "fft/SpatialConvolutionCuFFTTuner.h"
#include "cuda/KernelTimer.h"
#include "THC.h"
#include "fft/CuFFTConvolution.h"
#include "fft/SpatialConvolutionCuFFT.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <folly/Format.h>
#include <folly/Hash.h>
#include <folly/ScopeGuard.h>
#include <folly/Optional.h>
#include <unordered_map>

using namespace std;

namespace facebook { namespace deeplearning { namespace torch {

namespace {

typedef ProblemSizes::ProblemSizesTuple ProblemSizesTuple;

unordered_map<ProblemSizesTuple, CuFFTStrategy> strategyCache;

pair<string, CuFFTStrategy> getBestRun(ProblemSizes pbs) {
  // Atm, just focus on getting the bet perf, later trade this off with a
  // memory limit or flops per extra byte.
  string strategyMessage;
  folly::Optional<CuFFTStrategy> bestStrategy;
  auto bestTime = numeric_limits<float>::max();

  CuFFTStrategy s(pbs);
  auto funPass = (pbs.pass.pass == ConvolutionPass::kUpdateOutput) ?
    make_pair(detail::updateOutputTH,
              ConvolutionPass(ConvolutionPass::kUpdateOutput)) :
    (
      (pbs.pass.pass == ConvolutionPass::kUpdateGradInput) ?
      make_pair(detail::updateGradInputTH,
                ConvolutionPass(ConvolutionPass::kUpdateGradInput)) :
      make_pair(detail::accGradParametersTH,
                ConvolutionPass(ConvolutionPass::kAccGradParameters))
    );
  try {
    for (auto strat : s.makeStrategies()) {
      SCOPE_EXIT { detail::cleanupBuffers(); };
      THParams thp = strat.makeTensors();
      SCOPE_EXIT{ thp.free(); };

      size_t initialFree = 0;
      size_t totalFree = 0;
      THCudaCheck(cudaMemGetInfo(&initialFree, &totalFree));

      auto time = 0.0f;
      constexpr long kNumTrials = 5;
      for (int i = 0; i < kNumTrials; ++i) {
        cuda::KernelTimer timer;
        funPass.first(thp, pbs, strat);
        auto timeMS = timer.stop();
        if (i > 1) {
          time += timeMS;
        }
      }
      time /= kNumTrials - 2;

      stringstream ss;
      ss << strat;
      size_t finalFree = 0;
      THCudaCheck(cudaMemGetInfo(&finalFree, &totalFree));
      auto GOut = s.GReductions();

#define STRATEGY_MESSAGE                                                \
      folly::format(                                                    \
        "  Buffer={:.2f}M strategy = {} GReductions(virtual fmas)/s = {:.2f}" \
        "  time = {:.2f}ms",                                            \
        (initialFree - finalFree)/1e6, ss.str(), GOut / time * 1e3, time).str();

      // Log level 2, print all choices explored
      VLOG(2) << STRATEGY_MESSAGE;

      if (time > bestTime) {
        continue;
      }
      bestTime = time;
      bestStrategy = strat;
      strategyMessage = STRATEGY_MESSAGE;

#undef STRATEGY_MESSAGE

      // Log level 1, print the trace of best choices retained
      VLOG(1) << strategyMessage;
    }
  } catch (const bad_alloc& ex) {
    detail::cleanupBuffers();
    cudaDeviceReset();
    cudaFree(0);
    LOG(INFO) << ex.what();
  }

  CHECK(bestStrategy);
  return make_pair(std::move(strategyMessage), std::move(*bestStrategy));
}

void explorePerformance(
  ProblemSizes pbs, unordered_map<ProblemSizesTuple, CuFFTStrategy>& cache) {
  for (auto p : {
      ConvolutionPass(ConvolutionPass::kUpdateOutput),
        ConvolutionPass(ConvolutionPass::kUpdateGradInput),
        ConvolutionPass(ConvolutionPass::kAccGradParameters)}) {
    pbs = pbs.withPass(p);
    if (cache.count((ProblemSizesTuple)pbs) > 0) {
      continue;
    }
    { // Initial dump
      LOG(INFO) << folly::format(
        "START exploring FFT perf for pass = {}", p.toString()) << pbs;
    }
    auto res = getBestRun(pbs);
    cache.emplace((ProblemSizesTuple)pbs, res.second);
    LOG(INFO) << "Found best result " << res.first;
  }
}

} // namespace anon

CuFFTStrategy
SpatialConvolutionCuFFTTuner::getBestPerformance(ProblemSizes pbs) {
  if (strategyCache.count((ProblemSizesTuple)pbs) == 0) {
    // First function will explore for all passes at the current sizes
    explorePerformance(pbs, strategyCache);
  }
  CHECK_EQ(1, strategyCache.count((ProblemSizesTuple)pbs));
  return strategyCache[(ProblemSizesTuple)pbs];
}

}}}  // namespaces

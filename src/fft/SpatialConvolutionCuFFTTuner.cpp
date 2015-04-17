// Copyright 2014 Facebook

#include "SpatialConvolutionCuFFTTuner.h"

#include "cuda/KernelTimer.h"
#include "THC.h"
#include "CuFFTStrategy.h"
#include "SpatialConvolutionCuFFT.h"

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

unordered_map<ProblemSizesTuple, folly::Optional<CuFFTStrategy>> strategyCache;

folly::Optional<pair<string, CuFFTStrategy>>
getBestRun(THCState* state, ProblemSizes pbs) {
  // Atm, just focus on getting the best perf, later trade this off with a
  // memory limit or flops per extra byte.
  // `0` is cuFFT, `1` is fbfft best
  folly::Optional<string> strategyMessage[2];
  folly::Optional<CuFFTStrategy> bestStrategy[2];
  float bestTime[2] = {
    numeric_limits<float>::max(), numeric_limits<float>::max() };

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
      THParams thp = strat.makeTensors(state);
      SCOPE_EXIT{ thp.free(); };

      size_t initialFree = 0;
      size_t totalFree = 0;
      THCudaCheck(cudaMemGetInfo(&initialFree, &totalFree));

      // One pass through to stabilize the numbers
      funPass.first(state, thp, pbs, strat);

      auto time = 0.0f;
      constexpr auto kNumTrials = 1; // Numbers seem stable enough; we
                                     // can increase this if there is
                                     // a problem
      for (int i = 0; i < kNumTrials; ++i) {
        cuda::KernelTimer timer;
        funPass.first(state, thp, pbs, strat);
        time += timer.stop();
      }
      time /= (float) kNumTrials;

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

      auto ind = (strat.cufft()) ? 0 : 1;
      if (time > bestTime[ind]) {
        continue;
      }
      bestTime[ind] = time;
      bestStrategy[ind] = strat;
      strategyMessage[ind] = STRATEGY_MESSAGE;

#undef STRATEGY_MESSAGE

      // Log level 1, print the trace of best choices retained
      if (strategyMessage[0]) {
        VLOG(1) << *strategyMessage[0];
      }
      if (strategyMessage[1]) {
        VLOG(1) << *strategyMessage[1];
      }
    }
  } catch (const bad_alloc& ex) {
    detail::cleanupBuffers();
    cudaGetLastError(); // clear error state
    LOG(INFO) << ex.what();
  }

  if (strategyMessage[0]) {
    CHECK(bestStrategy[0]);
    LOG(INFO) << "Found best cufft result " << *strategyMessage[0];
  }
  if (strategyMessage[1]) {
    CHECK(bestStrategy[1]);
    LOG(INFO) << "Found best fbfft result " << *strategyMessage[1];
  }

  if (bestStrategy[0] && bestStrategy[1]) {
    return (bestTime[0] < bestTime[1]) ?
      make_pair(std::move(*strategyMessage[0]), std::move(*bestStrategy[0])) :
      make_pair(std::move(*strategyMessage[1]), std::move(*bestStrategy[1]));
  } else if (bestStrategy[0]) {
    return make_pair(std::move(*strategyMessage[0]),
                     std::move(*bestStrategy[0]));
  } else if (bestStrategy[1]) {
    return make_pair(std::move(*strategyMessage[1]),
                     std::move(*bestStrategy[1]));
  } else {
    // no solution
    return folly::none;
  }

  return (bestTime[0] < bestTime[1]) ?
    make_pair(std::move(*strategyMessage[0]), std::move(*bestStrategy[0])) :
    make_pair(std::move(*strategyMessage[1]), std::move(*bestStrategy[1]));
}

void explorePerformance(
  THCState* state,
  ProblemSizes pbs,
  unordered_map<ProblemSizesTuple, folly::Optional<CuFFTStrategy>>& cache) {
  for (auto p : {
      ConvolutionPass(ConvolutionPass::kUpdateOutput),
        ConvolutionPass(ConvolutionPass::kUpdateGradInput),
        ConvolutionPass(ConvolutionPass::kAccGradParameters)
        }
    ) {
    pbs = pbs.withPass(p);
    if (cache.count((ProblemSizesTuple) pbs)) {
      LOG(INFO) << "ALREADY GOT best result "
                << *cache[(ProblemSizesTuple) pbs];
      continue;
    }

    { // Initial dump
      LOG(INFO) << folly::format(
        "START exploring FFT perf for pass = {}", p.toString()) << pbs;
    }

    auto res = getBestRun(state, pbs);
    if (res) {
      LOG(INFO) << "Found best result " << res->first;
      cache.emplace((ProblemSizesTuple) pbs, std::move(res->second));
    } else {
      LOG(INFO) << "No solution possible";
      cache.emplace((ProblemSizesTuple) pbs, folly::none);
    }
  }
}

} // namespace anon

folly::Optional<CuFFTStrategy>
SpatialConvolutionCuFFTTuner::getBestPerformance(THCState* state,
                                                 ProblemSizes pbs) {
  if (!strategyCache.count((ProblemSizesTuple) pbs)) {
    // First function will explore for all passes at the current sizes
    explorePerformance(state, pbs, strategyCache);
  }

  // This will exist, as either a real solution or as folly::none
  CHECK(strategyCache.count((ProblemSizesTuple) pbs));
  return strategyCache[(ProblemSizesTuple) pbs];
}

}}}  // namespaces

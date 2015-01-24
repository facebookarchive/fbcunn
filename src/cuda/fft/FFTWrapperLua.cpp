// Copyright 2014 Facebook

#include "cuda/KernelTimer.h"
#include "DeviceTensorUtils.h"
#include "THC.h"
#include "THCTensor.h"
#include "fft/CuFFTWrapper.cuh"
#include "fft/FBFFT.h"
#include "util/Misc.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <folly/Format.h>
#include <glog/logging.h>
#include <luaT.h>
#include <lua.hpp>
#include <string>
#include <stdexcept>

using namespace facebook::cuda;
using namespace facebook::CUDAUtil;
using namespace std;

namespace facebook { namespace deeplearning { namespace torch {

namespace {

template <int Batch, int Dim>
float timedRun(THCudaTensor* timeTHTensor,
               THCudaTensor* frequencyTHTensor,
               FFTParameters p,
               cufftHandle fftPlan) {
  auto timeTensor =
    torchToDeviceTensor<float, Dim>(timeTHTensor);
  auto frequencyTensor =
    torchToDeviceTensor<float, Dim + 1>(frequencyTHTensor);
  if (p.cuFFT()) {
    if (fftPlan < 0) {
      fftPlan = makeCuFFTPlan<Batch, Dim>(timeTensor, frequencyTensor, p);
    }
    cuda::KernelTimer timer;
    fft<Batch, Dim>(timeTensor, frequencyTensor, p, &fftPlan);
    auto timeMS = timer.stop();
    return timeMS;
  } else {
    cuda::KernelTimer timer;
    auto result = fbfft<Batch, Dim>(timeTHTensor, frequencyTHTensor, p);
    if (result != FFTParameters::Success) {
      throw std::invalid_argument(folly::format("FBFFT error: {}",
                                                (int)result).str().c_str());
    }
    auto timeMS = timer.stop();
    return timeMS;
  }
  return 0.0f;
}

#define FFT_BATCH(BATCH)                                        \
  case BATCH:                                                   \
  {                                                             \
    switch(dims) {                                              \
      case 2:                                                   \
        time += timedRun<BATCH, 2>(timeTHTensor,                \
                                   frequencyTHTensor,           \
                                   p,                           \
                                   fftPlan);                    \
        break;                                                  \
      case 3:                                                   \
        time += timedRun<BATCH, 3>(timeTHTensor,                \
                                   frequencyTHTensor,           \
                                   p,                           \
                                   fftPlan);                    \
        break;                                                  \
      default:                                                  \
        throw invalid_argument("Unsupported dims + batchDims"); \
    }                                                           \
  }                                                             \
  break;



int fftFun(lua_State* L, bool forward) {
  bool dumpTimings = false;

  auto batchDims = luaT_getfieldcheckint(L, 1, "batchDims");
  auto cufft = luaT_getfieldcheckint(L, 1, "cufft");
  auto timeTHTensor =
    (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  auto frequencyTHTensor =
    (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");

  CHECK_EQ(THCudaTensor_nDimension(NULL, timeTHTensor) + 1,
           THCudaTensor_nDimension(NULL, frequencyTHTensor));

  auto time = 0.0f;
  constexpr int kNumTrials = 10;
  int dims = THCudaTensor_nDimension(NULL, timeTHTensor);
  FFTParameters p; // forward and normalize are default
  if (!forward) {
    p = p.inverse().normalize(false);
  }
  if (cufft == 1) {
    p = p.withCufft();
  } else {
    p = p.withFbfft();
  }

  try {
    cufftHandle fftPlan = -1;
    SCOPE_EXIT{
      if (fftPlan >= 0) {
        CHECK_EQ(CUFFT_SUCCESS, cufftDestroy(fftPlan));
      }
    };

    for (int i = 0; i < kNumTrials; ++i) {
      switch (batchDims) {
        FFT_BATCH(1);
        // FFT_BATCH(2);
        // FFT_BATCH(3);
        default:
          throw invalid_argument("Unsupported batch dims");
      };

      // Reset time to kNumTrials
      if (i == 0 && kNumTrials > 1) {
        time = 0.0f;
      }
    }
    time /= std::max(1, kNumTrials - 1);
  } catch(exception &e){
    return luaL_error(L, e.what());
  }

  long batches = 1;
  for (auto i = 0; i < batchDims; ++i) {
    batches *= THCudaTensor_size(NULL, timeTHTensor, i);
  }

  // 1-D -> batches * N log N
  // 2-D -> batches * (M N log N + N M logM)
  // 3-D -> batches * (M N P log P + P N M logM +  M P N logN)
  float size = batches;
  for (int i = 1; i < dims; ++i){
    size *= THCudaTensor_size(NULL, timeTHTensor, dims - i);
  }
  float logs = 1.0f;
  for (int i = 1; i < dims; ++i){
    logs += log(THCudaTensor_size(NULL, timeTHTensor, dims - i));
  }
  size *= logs;

  auto version = (p.cuFFT()) ? "CuFFT" :
    ((p.nyuFFT()) ? "NYUFFT" : "FBFFT");
  auto direction = (forward) ? "forward" : "inverse";
  auto GOut = size / 1e9;
  LOG_IF(INFO, dumpTimings) << folly::format(
    "  Running fft-{}d ({}) direction={} ({}x{}x{}),"   \
    "  {} batches, GNlogN/s = {:.5f}"                   \
    "  time = {:.2f}ms",
    dims - batchDims,
    version,
    direction,
    (dims >= 1) ? THCudaTensor_size(NULL, timeTHTensor, 0) : 1,
    (dims >= 2) ? THCudaTensor_size(NULL, timeTHTensor, 1) : 1,
    (dims >= 3) ? THCudaTensor_size(NULL, timeTHTensor, 2) : 1,
    batches,
    (GOut / time) * 1e3,
    time).str();

  return 0;
}

int fftFun(lua_State* L) {
  return fftFun(L, true);
}

int fftiFun(lua_State* L) {
  return fftFun(L, false);
}

const luaL_Reg functions[] = {
  {"FFTWrapper_fft", fftFun},
  {"FFTWrapper_ffti", fftiFun},
  {nullptr, nullptr},
};

}  // namespace

void initFFTWrapper(lua_State* L) {
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, functions, "nn");
  lua_pop(L, 1);
}

}}}  // namespaces

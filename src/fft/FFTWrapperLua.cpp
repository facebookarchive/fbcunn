// Copyright 2014 Facebook

#include "cuda/KernelTimer.h"

#include "THC.h"
#include "THCTensor.h"
#include "cuda/fbfft/FBFFT.cuh"
#include "cuda/util/CachedDeviceProperties.h"
#include "src/Utils.h"
#include "src/fft/CuFFTWrapper.cuh"
#include "src/fft/FBFFTHost.h"
#include "src/DeviceTensorUtils.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <folly/Format.h>
#include <glog/logging.h>
#include <luaT.h>
#include <lua.hpp>
#include <string>
#include <stdexcept>

using namespace facebook::cuda;
using namespace facebook::cuda::fbfft;
using namespace std;

namespace facebook { namespace deeplearning { namespace torch {

namespace {

template <int Batch, int Dim>
float timedRun(THCState* state,
               THCudaTensor* timeTHTensor,
               THCudaTensor* frequencyTHTensor,
               THCudaTensor* bufferTHTensor,
               FFTParameters p,
               cufftHandle fftPlan) {
  cufftHandle localPlan = fftPlan;
  SCOPE_EXIT {
    if (fftPlan < 0) {
      cufftDestroy(localPlan);
    }
  };
  auto timeTensor =
    torchToDeviceTensor<float, Dim>(state, timeTHTensor);
  auto frequencyTensor =
    torchToDeviceTensor<float, Dim + 1>(state, frequencyTHTensor);
  if (p.cuFFT()) {
    if (fftPlan < 0) {
      localPlan = makeCuFFTPlan<Batch, Dim>(timeTensor, frequencyTensor, p);
    }
    cuda::KernelTimer timer;
    fft<Batch, Dim>(timeTensor, frequencyTensor, p, &localPlan);
    auto timeMS = timer.stop();
    return timeMS;
  } else {
    cuda::KernelTimer timer;
    auto result = fbfft<Batch>(
      state,
      timeTHTensor, frequencyTHTensor, bufferTHTensor, (FBFFTParameters)p);
    if (result != FBFFTParameters::Success) {
      THCudaCheck(cudaGetLastError());
      THError(folly::format("FBFFT error: {}", (int)result).str().c_str());
    }
    auto timeMS = timer.stop();
    return timeMS;
  }
  return 0.0f;
}

#define TIMED_FFT(BATCH, DIM)                           \
  if (batchDims == BATCH && dims == DIM) {              \
    time += timedRun<BATCH, DIM>(state,                 \
                                 timeTHTensor,          \
                                 frequencyTHTensor,     \
                                 bufferTHTensor,        \
                                 p,                     \
                                 fftPlan);              \
    done = true;                                        \
  }

int runTimedFFT(lua_State* L, bool forward) {
  THCState* state = getCutorchState(L);
  auto batchDims = luaT_getfieldcheckint(L, 1, "batchDims");
  auto cufft = luaT_getfieldcheckboolean(L, 1, "cufft");
  auto padLeft = luaT_getfieldcheckint(L, 1, "padLeft");
  auto padUp = luaT_getfieldcheckint(L, 1, "padUp");
  auto timeTHTensor =
    (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  auto frequencyTHTensor =
    (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
  auto bufferTHTensor =
    (THCudaTensor*)luaT_checkudata(L, 4, "torch.CudaTensor");
  if (THCudaTensor_nDimension(state, bufferTHTensor) == 0) {
    bufferTHTensor = nullptr;
    THAssert(THCudaTensor_checkGPU(state, 2, timeTHTensor, frequencyTHTensor));
  } else {
    THAssert(THCudaTensor_checkGPU(state, 3, timeTHTensor, frequencyTHTensor,
                                   bufferTHTensor));
  }
  auto fftPlan = (cufftHandle)lua_tonumber(L, 5);

  CHECK_EQ(THCudaTensor_nDimension(state, timeTHTensor) + 1,
           THCudaTensor_nDimension(state, frequencyTHTensor));

  auto time = 0.0f;
  constexpr int kNumTrials = 2;
  constexpr int kNumSkipTrials = 1;
  int dims = THCudaTensor_nDimension(state, timeTHTensor);
  FFTParameters p; // forward and normalize are default
  if (!forward) {
    p = p.inverse().normalize(false);
  }
  if (cufft) {
    p = p.withCufft();
  } else {
    p = p.withFbfft();
  }
  p.withPadLeft(padLeft);
  p.withPadUp(padUp);

  try {
    for (int i = 0; i < kNumTrials; ++i) {
      auto done = false;
      TIMED_FFT(1, 2);
      TIMED_FFT(1, 3);
      if (!done) {
        THCudaCheck(cudaGetLastError());
        THError("Timed FFT: Unsupported batch dims");
      }
      // Reset time to kNumTrials
      if (i < kNumSkipTrials && kNumTrials > kNumSkipTrials) {
        time = 0.0f;
      }
    }
    time /= std::max(1, kNumTrials - kNumSkipTrials);
  } catch(exception &e){
    return luaL_error(L, e.what());
  }

  long batches = 1;
  for (auto i = 0; i < batchDims; ++i) {
    batches *= THCudaTensor_size(state, timeTHTensor, i);
  }

  // 1-D -> batches * N log N
  // 2-D -> batches * (M N log N + N M logM)
  // 3-D -> batches * (M N P log P + P N M logM +  M P N logN)
  float size = batches;
  for (int i = 1; i < dims; ++i){
    size *= THCudaTensor_size(state, timeTHTensor, dims - i);
  }
  float logs = 1.0f;
  for (int i = 1; i < dims; ++i){
    logs += log(THCudaTensor_size(state, timeTHTensor, dims - i));
  }
  size *= logs;

  auto version = (p.cuFFT()) ? "CuFFT" : "FBFFT";
  auto direction = (forward) ? "forward" : "inverse";
  auto GOut = size / 1e9;
  LOG(INFO) << folly::format(
    "  Running fft-{}d ({}) direction={} ({}x{}x{}),"   \
    "  {} batches, GNlogN/s = {:.5f}"                   \
    "  time = {:.2f}ms",
    dims - batchDims,
    version,
    direction,
    (dims >= 1) ? THCudaTensor_size(state, timeTHTensor, 0) : 1,
    (dims >= 2) ? THCudaTensor_size(state, timeTHTensor, 1) : 1,
    (dims >= 3) ? THCudaTensor_size(state, timeTHTensor, 2) : 1,
    batches,
    (GOut / time) * 1e3,
    time).str();

  return 0;
}

#define FBFFT_CASE(BATCH_DIMS, INPUT_DIMS)                              \
  if (batchDims == BATCH_DIMS && inputDims == INPUT_DIMS) {             \
    auto result = fbfft<BATCH_DIMS>(state,                              \
                                    timeTHTensor,                       \
                                    frequencyTHTensor,                  \
                                    bufferTHTensor,                     \
                                    (FBFFTParameters)p);                \
    if (result != FBFFTParameters::Success) {                           \
      THCudaCheck(cudaGetLastError());                                  \
      THError(                                                          \
        folly::format("FBFFT error: {}",                                \
                      (int)result).str().c_str());                      \
    }                                                                   \
    done = true;                                                        \
  }

#define CUFFT_CASE(BATCH_DIMS, INPUT_DIMS)                                  \
  if (batchDims == BATCH_DIMS && inputDims == INPUT_DIMS) {                 \
    auto timeTensor =                                                       \
      torchToDeviceTensor<float, INPUT_DIMS>(state, timeTHTensor);          \
    auto frequencyTensor =                                                  \
      torchToDeviceTensor<float, INPUT_DIMS + 1>(state, frequencyTHTensor); \
    if (fftPlan < 0) {                                                      \
       localPlan = makeCuFFTPlan<BATCH_DIMS, INPUT_DIMS>(                   \
         timeTensor, frequencyTensor, p);                                   \
    }                                                                       \
    fft<BATCH_DIMS, INPUT_DIMS>(timeTensor, frequencyTensor, p, &localPlan);\
    done = true;                                                            \
  }

int runFFT(lua_State* L, bool forward) {
  THCState* state = getCutorchState(L);
  auto batchDims = luaT_getfieldcheckint(L, 1, "batchDims");
  auto cufft = luaT_getfieldcheckboolean(L, 1, "cufft");
  auto padLeft = luaT_getfieldcheckint(L, 1, "padLeft");
  auto padUp = luaT_getfieldcheckint(L, 1, "padUp");
  auto timeTHTensor =
    (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  auto frequencyTHTensor =
    (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
  auto bufferTHTensor =
    (THCudaTensor*)luaT_checkudata(L, 4, "torch.CudaTensor");
  if (THCudaTensor_nDimension(state, bufferTHTensor) == 0) {
    bufferTHTensor = nullptr;
    THAssert(THCudaTensor_checkGPU(state, 2, timeTHTensor, frequencyTHTensor));
  } else {
    THAssert(THCudaTensor_checkGPU(state, 3, timeTHTensor, frequencyTHTensor,
                                   bufferTHTensor));
  }
  auto fftPlan = (cufftHandle)lua_tonumber(L, 5);

  CHECK_EQ(THCudaTensor_nDimension(state, timeTHTensor) + 1,
           THCudaTensor_nDimension(state, frequencyTHTensor));

  int inputDims = THCudaTensor_nDimension(state, timeTHTensor);
  FFTParameters p; // forward and normalize are default
  if (!forward) {
    p = p.inverse().normalize(false);
  }
  if (!cufft) {
    p = p.withFbfft();
  } else {
    p = p.withCufft();
  }
  p.withPadLeft(padLeft);
  p.withPadUp(padUp);

  try {
    auto done = false;
    if (!cufft) {
      FBFFT_CASE(1, 2);
      FBFFT_CASE(1, 3);
      FBFFT_CASE(2, 3);
      FBFFT_CASE(2, 4);
      if (!done) { THError("Unsupported fbfft batch dims"); }
    } else {
      cufftHandle localPlan = fftPlan;
      SCOPE_EXIT {
        if (fftPlan < 0) {
          cufftDestroy(localPlan);
        }
      };
      CUFFT_CASE(1, 2);
      CUFFT_CASE(1, 3);
      CUFFT_CASE(2, 3);
      CUFFT_CASE(2, 4);
      if (!done) { THError("Unsupported cufft batch dims"); }
    }
  } catch(exception &e){
    return luaL_error(L, e.what());
  }

  return 0;
}

int fft(lua_State* L) {
  auto timed = luaT_getfieldcheckboolean(L, 1, "timed");
  if (timed) { return runTimedFFT(L, true); }
  return runFFT(L, true);
}

int ffti(lua_State* L) {
  auto timed = luaT_getfieldcheckboolean(L, 1, "timed");
  if (timed) { return runTimedFFT(L, false); }
  return runFFT(L, false);
}

const luaL_Reg functions[] = {
  {"FFTWrapper_fft", fft},
  {"FFTWrapper_ffti", ffti},
  {nullptr, nullptr},
};

}  // namespace

void initFFTWrapper(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, functions, "nn");
  lua_pop(L,1);
}

}}}  // namespaces

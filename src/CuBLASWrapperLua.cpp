// Copyright 2014 Facebook

#include "cuda/KernelTimer.h"
#include "Utils.h"
#include "DeviceTensorUtils.h"
#include "THC.h"
#include "THCTensor.h"
#include "CuBLASWrapper.h"
#include "util/Misc.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <folly/Format.h>
#include <glog/logging.h>
#include <luaT.h>
#include <lua.hpp>
#include <string>

using namespace facebook::cuda;
using namespace facebook::CUDAUtil;
using namespace std;

namespace facebook { namespace deeplearning { namespace torch {

namespace {

#define MATMULT_CASE(DIM)                                               \
  case DIM:                                                             \
  CHECK_EQ(DIM, iterDims + batchDims + 2);                              \
  {                                                                     \
    DeviceTensor<float, DIM> A = torchToDeviceTensor<float, DIM>(state, thA);  \
    DeviceTensor<float, DIM> B = torchToDeviceTensor<float, DIM>(state, thB);  \
    DeviceTensor<float, DIM> C = torchToDeviceTensor<float, DIM>(state, thC);  \
    matmultIter<DIM>(C, A, B, params);                                  \
  }                                                                     \
  break;

int matmult(lua_State* L, bool asComplex = false) {
  THCState* state = getCutorchState(L);
  auto iterDims = luaT_getfieldcheckint(L, 1, "iterDims");
  auto batchDims = luaT_getfieldcheckint(L, 1, "batchDims");
  auto numHandles = luaT_getfieldcheckint(L, 1, "handles");
  auto numStreams = luaT_getfieldcheckint(L, 1, "streams");
  auto thA = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  auto thB = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
  auto thC = (THCudaTensor*)luaT_checkudata(L, 4, "torch.CudaTensor");

  THAssert(THCudaTensor_checkGPU(state, 3, thA, thB, thC));

  CHECK_EQ(THCudaTensor_nDimension(state, thA),
           THCudaTensor_nDimension(state, thB));
  CHECK_EQ(THCudaTensor_nDimension(state, thC),
           THCudaTensor_nDimension(state, thB));

  std::vector<cublasHandle_t> handles;
  for (auto i = 0; i < numHandles; ++i) {
    handles.push_back(cublasHandle_t());
    cublasCreate(&(handles.back()));
  }

  std::vector<cudaStream_t> streams;
  for (auto i = 0; i < numStreams; ++i) {
    streams.push_back(cudaStream_t());
    cudaStreamCreate(&(streams.back()));
  }

  auto time = 0.0f;
  constexpr long kNumTrials = 5;
  int dims = THCudaTensor_nDimension(state, thA);
  BLASParameters p;
  auto& params = p.withIterDims(iterDims).withBatchDims(batchDims).
    withComplex(asComplex).withHandles(handles).withStreams(streams);
  for (int i = 0; i < kNumTrials; ++i) {
    cuda::KernelTimer timer;
    switch (dims) {
      MATMULT_CASE(2);
      MATMULT_CASE(3);
      MATMULT_CASE(4);
      MATMULT_CASE(5);
      MATMULT_CASE(6);
      default:
        throw invalid_argument("Unsupported dims");
    };
    auto timeMS = timer.stop();
    if (i > 0) {
      time += timeMS;
    }
  }
  time /= kNumTrials - 1;

  long iters = 1;
  for (int i = 0; i < iterDims; ++i) {
    iters *= THCudaTensor_size(state, thA, i);
  }
  long batch = 1;
  for (int i = iterDims; i < iterDims + batchDims; ++i) {
    batch *= THCudaTensor_size(state, thA, i);
  }

  auto GOut = (THCudaTensor_size(state, thC, 0) *
               THCudaTensor_stride(state, thC, 0) *
               THCudaTensor_size(state, thA, dims - 1)) /
               1e9;
  LOG(INFO) << folly::format(
    "  Running mxm ({}x{}x{}): {} iterations (parallel over streams),"  \
    "  {} batches, GReductions(virtual fmas)/s = {:.5f}"                \
    "  time = {:.2f}ms",
    THCudaTensor_size(state, thC, dims - 2),
    THCudaTensor_size(state, thC, dims - 1),
    THCudaTensor_size(state, thA, dims - 1),
    iters,
    batch,
    (GOut / time) * 1e3,
    time).str();

  return 0;
}

int matmult(lua_State* L) {
  return matmult(L, false);
}

int matmultComplex(lua_State* L) {
  return matmult(L, true);
}

const luaL_Reg functions[] = {
  {"CuBLASWrapper_matmult", matmult},
  {"CuBLASWrapper_matmultComplex", matmultComplex},
  {nullptr, nullptr},
};

}  // namespace

void initCuBLASWrapper(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, functions, "nn");
  lua_pop(L,1);
}

}}}  // namespaces

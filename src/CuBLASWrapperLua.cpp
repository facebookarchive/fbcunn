// Copyright 2014 Facebook

#include "cuda/KernelTimer.h"
#include "cuda/util/CachedDeviceProperties.h"
#include "src/Utils.h"
#include "src/DeviceTensorUtils.h"
#include "THC.h"
#include "THCTensor.h"
#include "src/CuBLASWrapper.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <folly/Format.h>
#include <glog/logging.h>
#include <luaT.h>
#include <lua.hpp>
#include <string>

using namespace facebook::cuda;
using namespace std;

namespace facebook { namespace deeplearning { namespace torch {

namespace {

#define LOG_TARGET VLOG(3)

#define MATMULT_CASE(DIM)                                                     \
  case DIM:                                                                   \
  CHECK_EQ(DIM, iterDims + batchDims + 2 + ((asComplex) ? 1 : 0));            \
  {                                                                           \
    DeviceTensor<float, DIM> A = torchToDeviceTensor<float, DIM>(state, thA); \
    DeviceTensor<float, DIM> B = torchToDeviceTensor<float, DIM>(state, thB); \
    DeviceTensor<float, DIM> C = torchToDeviceTensor<float, DIM>(state, thC); \
    matmultIter<DIM>(C, A, B, params);                                        \
  }                                                                           \
  break;

int matmult(lua_State* L, bool asComplex = false) {
  THCState* state = getCutorchState(L);
  auto transA = luaT_getfieldcheckstring(L, 1, "transA");
  auto transB = luaT_getfieldcheckstring(L, 1, "transB");
  auto iterDims = luaT_getfieldcheckint(L, 1, "iterDims");
  auto batchDims = luaT_getfieldcheckint(L, 1, "batchDims");
  auto scale = luaT_getfieldchecknumber(L, 1, "scale");
  auto timed = luaT_getfieldcheckboolean(L, 1, "timed");
  auto thA = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  auto thB = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
  auto thC = (THCudaTensor*)luaT_checkudata(L, 4, "torch.CudaTensor");

  THAssert(THCudaTensor_checkGPU(state, 3, thA, thB, thC));

  CHECK_EQ(THCudaTensor_nDimension(state, thA),
           THCudaTensor_nDimension(state, thB));
  CHECK_EQ(THCudaTensor_nDimension(state, thC),
           THCudaTensor_nDimension(state, thB));

  int device;
  THCudaCheck(cudaGetDevice(&device));

  std::vector<cublasHandle_t> handles;
  // Skip NULL handle
  for (auto i = 1; i <= THCState_getNumBlasHandles(state); ++i) {
    handles.push_back(THCState_getDeviceBlasHandle(state, device, i));
  }

  std::vector<cudaStream_t> streams;
  // Skip default stream
  for (auto i = 1; i <= THCState_getNumStreams(state); ++i) {
    streams.push_back(THCState_getDeviceStream(state, device, i));
  }

  int dims = THCudaTensor_nDimension(state, thA);
  BLASParameters p;
  auto& params = p.withIterDims(iterDims).withBatchDims(batchDims).
    withComplex(asComplex).withHandles(handles).withStreams(streams).
    withTransposeA(transA[0]).withTransposeB(transB[0]).withScaleReal(scale);

  if (!timed) {
    switch (dims) {
      MATMULT_CASE(2);
      MATMULT_CASE(3);
      MATMULT_CASE(4);
      MATMULT_CASE(5);
      MATMULT_CASE(6);
      default:
        THError("GEMM Unsupported dims");
    };
  } else {
    auto time = 0.0f;
    constexpr long kNumTrials = 5;
    for (int i = 0; i < kNumTrials; ++i) {
      cuda::KernelTimer timer;
      switch (dims) {
        MATMULT_CASE(2);
        MATMULT_CASE(3);
        MATMULT_CASE(4);
        MATMULT_CASE(5);
        MATMULT_CASE(6);
        default:
          THError("GEMM Unsupported dims");
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
    LOG_TARGET << folly::format(
      "  Running mxm ({}x{}x{}): {} iterations (parallel over streams),"  \
      "  {} batches, GReductions(virtual fmas)/s = {:.5f}"                \
      "  time = {:.2f}ms",
      THCudaTensor_size(state, thC, (asComplex) ? dims - 3 : dims - 2),
      THCudaTensor_size(state, thC, (asComplex) ? dims - 2 : dims - 1),
      THCudaTensor_size(state, thA, (asComplex) ? dims - 2 : dims - 1),
      iters,
      batch,
      (GOut / time) * 1e3,
      time).str();
  }

  return 0;
}

int matmult(lua_State* L) {
  return matmult(L, false);
}

int matmultComplex(lua_State* L) {
  return matmult(L, true);
}

#define TRANSPOSE_CASE(DIM)                                                    \
  if (dim == DIM) {                                                            \
    DeviceTensor<float, DIM> A = torchToDeviceTensor<float, DIM>(state, thA);  \
    DeviceTensor<float, DIM> tA = torchToDeviceTensor<float, DIM>(state, thB); \
    facebook::deeplearning::torch::transpose<DIM>(                             \
      A, tA, separator, asComplex, transposeMetaData, handle, stream);         \
    if (transposeMetaData) {                                                   \
      /* Also transpose the metadata */                                        \
      for (auto i = 0; i < dim; ++i) {                                         \
        thB->size[i] = tA.getSize(i);                                          \
        thB->stride[i] = tA.getStride(i);                                      \
      }                                                                        \
    }                                                                          \
    done = true;                                                               \
  }

int transpose(lua_State* L, bool asComplex = false) {
  THCState* state = getCutorchState(L);
  auto separator  = luaT_getfieldcheckint(L, 1, "separator");
  auto transposeMetaData = luaT_getfieldcheckboolean(L, 1, "transposeMetaData");
  auto handleIndex = luaT_getfieldcheckint(L, 1, "handle");
  auto streamIndex = luaT_getfieldcheckint(L, 1, "stream");
  auto thA = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  auto thB = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
  int dim = THCudaTensor_nDimension(state, thA);

  CHECK_EQ(THCudaTensor_nDimension(state, thA),
           THCudaTensor_nDimension(state, thB));

  int device;
  THCudaCheck(cudaGetDevice(&device));

  auto handle = THCState_getDeviceBlasHandle(state, device, handleIndex);
  auto stream = THCState_getDeviceStream(state, device, streamIndex);

  auto done = false;
  TRANSPOSE_CASE(2);
  TRANSPOSE_CASE(3);
  TRANSPOSE_CASE(4);
  TRANSPOSE_CASE(5);
  if (!done) { THError("Transpose Unsupported dims"); }

  return 0;
}

int transpose(lua_State* L) {
  return transpose(L, false);
}

int transposeComplex(lua_State* L) {
  return transpose(L, true);
}

const luaL_Reg functions[] = {
  {"CuBLASWrapper_matmult", matmult},
  {"CuBLASWrapper_matmultComplex", matmultComplex},
  {"CuBLASWrapper_transpose", transpose},
  {"CuBLASWrapper_transposeComplex", transposeComplex},
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

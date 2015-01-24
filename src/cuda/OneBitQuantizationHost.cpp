// Copyright 2014 Facebook

#include "cuda/DeviceTensor.cuh"
#include "DeviceTensorUtils.h"
#include "THC.h"
#include "THCTensor.h"
#include "OneBitQuantization.cuh"

#include <cuda_runtime.h>
#include <glog/logging.h>
#include <luaT.h>
#include <lua.hpp>

using namespace std;
using namespace facebook::cuda;

namespace facebook { namespace deeplearning { namespace torch {

namespace {

constexpr int kNumBits = sizeof(unsigned) * 8;

constexpr int toQuantizedSize(int size) {
  return (size + kNumBits - 1) / kNumBits;
}

int quantize(lua_State *L) {
  auto nonQuantizedTH = (THCudaTensor*)luaT_checkudata(
    L, 2, "torch.CudaTensor");
  auto quantizedTH = (THCudaTensor*)luaT_getfieldcheckudata(
    L, 1, "quantized", "torch.CudaTensor");
  auto quantizationErrorTH = (THCudaTensor*)luaT_getfieldcheckudata(
    L, 1, "quantization_error", "torch.CudaTensor");
  auto avgPosTH = (THCudaTensor*)luaT_getfieldcheckudata(
    L, 1, "avg_pos", "torch.CudaTensor");
  auto avgNegTH = (THCudaTensor*)luaT_getfieldcheckudata(
    L, 1, "avg_neg", "torch.CudaTensor");

  // The input should be two-dimensional
  luaL_argcheck(L, THCudaTensor_nDimension(NULL, nonQuantizedTH) == 2, 2,
                "non_quantized_input should be 2d");

  const auto rows = THCudaTensor_size(NULL, nonQuantizedTH, 0);
  const auto cols = THCudaTensor_size(NULL, nonQuantizedTH, 1);

  // Make sure that the outputs are properly sized
  THCudaTensor_resize2d(NULL, quantizedTH, rows, toQuantizedSize(cols));
  THCudaTensor_resize2d(NULL, quantizationErrorTH, rows, cols);
  THCudaTensor_resize1d(NULL, avgPosTH, rows);
  THCudaTensor_resize1d(NULL, avgNegTH, rows);

  DeviceTensor<float, 2> nonQuantized =
    torchToDeviceTensor<float, 2>(nonQuantizedTH);
  DeviceTensor<float, 2> quantized =
    torchToDeviceTensor<float, 2>(quantizedTH);
  DeviceTensor<float, 2> quantizationError =
    torchToDeviceTensor<float, 2>(quantizationErrorTH);
  DeviceTensor<float, 1> avgPos =
    torchToDeviceTensor<float, 1>(avgPosTH);
  DeviceTensor<float, 1> avgNeg =
    torchToDeviceTensor<float, 1>(avgNegTH);

  runQuantize1Bit(nonQuantized, quantized, quantizationError, avgPos, avgNeg);

  return 0;
}

int dequantize(lua_State *L) {
  auto quantizedTH = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  auto avgPosTH = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
  auto avgNegTH = (THCudaTensor*)luaT_checkudata(L, 4, "torch.CudaTensor");
  auto nonQuantizedCols = luaL_checkint(L, 5);
  auto nonQuantizedTH = (THCudaTensor*)luaT_getfieldcheckudata(
    L, 1, "non_quantized", "torch.CudaTensor");

  // The input should be two-dimensional
  luaL_argcheck(L, THCudaTensor_nDimension(NULL, quantizedTH) == 2, 2,
                "input should be 2d");

  const auto rows = THCudaTensor_size(NULL, quantizedTH, 0);
  const auto quantizedCols = THCudaTensor_size(NULL, quantizedTH, 1);

  // The input should be within appropriate quantization sizes
  luaL_argcheck(L, quantizedCols == toQuantizedSize(nonQuantizedCols), 5,
                "num_orig_cols does not match quantized_input cols");
  luaL_argcheck(L, THCudaTensor_size(NULL, avgPosTH, 0) == rows, 3,
                "avg_pos size doesn't match quantized_input rows");
  luaL_argcheck(L, THCudaTensor_size(NULL, avgNegTH, 0) == rows, 4,
                "avg_neg size doesn't match quantized_input rows");

  // Make sure that the outputs are properly sized
  THCudaTensor_resize2d(NULL, nonQuantizedTH, rows, nonQuantizedCols);

  DeviceTensor<float, 2> quantized =
    torchToDeviceTensor<float, 2>(quantizedTH);
  DeviceTensor<float, 1> avgPos =
    torchToDeviceTensor<float, 1>(avgPosTH);
  DeviceTensor<float, 1> avgNeg =
    torchToDeviceTensor<float, 1>(avgNegTH);
  DeviceTensor<float, 2> nonQuantized =
    torchToDeviceTensor<float, 2>(nonQuantizedTH);

  runDequantize1Bit(quantized, avgPos, avgNeg, nonQuantized);

  return 0;
}

const luaL_Reg functions [] = {
  {"OneBitQuantization_quantize", quantize},
  {"OneBitQuantization_dequantize", dequantize},
  {nullptr, nullptr}
};

}  // namespace

void initOneBitQuantizationCuda(lua_State *L) {
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, functions, "nn");
  lua_pop(L,1);
}

}}}  // namespaces

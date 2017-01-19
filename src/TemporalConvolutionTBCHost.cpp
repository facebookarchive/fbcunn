// Copyright 2016 Facebook

#include "THC.h"
#include "THCTensor.h"
#include "cuda/DeviceTensor.cuh"
#include "src/DeviceTensorUtils.h"
#include "src/TemporalConvolutionTBC.cuh"
#include "src/Utils.h"

#include <cuda_runtime.h>
#include <glog/logging.h>
#include <lua.hpp>
#include <luaT.h>

using namespace std;
using namespace facebook::cuda;

namespace facebook {
namespace deeplearning {
namespace torch {

namespace {

inline THCudaTensor*
getFieldCudaTensor(lua_State* L, int arg, const char* name) {
  return static_cast<THCudaTensor*>(
      luaT_getfieldcheckudata(L, arg, name, "torch.CudaTensor"));
}
inline THCudaTensor* getCudaTensor(lua_State* L, int arg) {
  return static_cast<THCudaTensor*>(
      luaT_checkudata(L, arg, "torch.CudaTensor"));
}

int updateOutput(lua_State* L) {
  THCState* state = getCutorchState(L);
  auto output = (THCudaTensor*)luaT_getfieldcheckudata(
      L, 1, "output", "torch.CudaTensor");
  auto weight = (THCudaTensor*)luaT_getfieldcheckudata(
      L, 1, "weight", "torch.CudaTensor");
  auto bias =
      (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "bias", "torch.CudaTensor");
  auto input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");

  THAssert(THCudaTensor_checkGPU(state, 4, input, output, weight, bias));

  auto inputDev = torchToDeviceTensor<float, 3>(state, input);
  auto outputDev = torchToDeviceTensor<float, 3>(state, output);
  auto weightDev = torchToDeviceTensor<float, 3>(state, weight);
  auto biasDev = torchToDeviceTensor<float, 1>(state, bias);

  detail::runTemporalConvolutionTBC_updateOutput(
      state, inputDev, outputDev, weightDev, biasDev);

  return 0;
}

int updateGradInput(lua_State* L) {
  THCState* state = getCutorchState(L);
  auto dInput = (THCudaTensor*)luaT_getfieldcheckudata(
      L, 1, "gradInput", "torch.CudaTensor");
  auto weight = (THCudaTensor*)luaT_getfieldcheckudata(
      L, 1, "weight", "torch.CudaTensor");
  auto dOutput = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");

  THAssert(THCudaTensor_checkGPU(state, 3, dInput, dOutput, weight));

  auto dInputDev = torchToDeviceTensor<float, 3>(state, dInput);
  auto dOutputDev = torchToDeviceTensor<float, 3>(state, dOutput);
  auto weightDev = torchToDeviceTensor<float, 3>(state, weight);

  detail::runTemporalConvolutionTBC_updateGradInput(
      state, dInputDev, dOutputDev, weightDev);

  return 0;
}

int accGradParameters(lua_State* L) {
  THCState* state = getCutorchState(L);
  auto dWeight = (THCudaTensor*)luaT_getfieldcheckudata(
      L, 1, "gradWeight", "torch.CudaTensor");
  auto dBias = (THCudaTensor*)luaT_getfieldcheckudata(
      L, 1, "gradBias", "torch.CudaTensor");
  auto input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  auto dOutput = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
  float scale = lua_tonumber(L, 4);

  THAssert(THCudaTensor_checkGPU(state, 4, input, dOutput, dWeight, dBias));

  auto inputDev = torchToDeviceTensor<float, 3>(state, input);
  auto dOutputDev = torchToDeviceTensor<float, 3>(state, dOutput);
  auto dWeightDev = torchToDeviceTensor<float, 3>(state, dWeight);
  auto dBiasDev = torchToDeviceTensor<float, 1>(state, dBias);

  detail::runTemporalConvolutionTBC_accGradParameters(
      state, inputDev, dOutputDev, dWeightDev, dBiasDev, scale);

  return 0;
}

const luaL_Reg functions[] = {
    {"TemporalConvolutionTBC_updateOutput", updateOutput},
    {"TemporalConvolutionTBC_updateGradInput", updateGradInput},
    {"TemporalConvolutionTBC_accGradParameters", accGradParameters},
    {nullptr, nullptr}};

} // namespace

void initTemporalConvolutionTBCCuda(lua_State* L) {
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, functions, "nn");
  lua_pop(L, 1);
}
}
}
} // namespaces

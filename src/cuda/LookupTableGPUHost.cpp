/**
 * Copyright 2014 Facebook
 * @author Michael Mathieu (myrhev@fb.com)
 */

#include "cuda/DeviceTensor.cuh"
#include "DeviceTensorUtils.h"
#include "THC.h"

#include <lua.hpp>
#include <TH.h>
#include <luaT.h>

using namespace facebook::cuda;

namespace facebook { namespace deeplearning { namespace torch {

namespace detail {
void launchLookupTableGPUUpdateOutputKernel(
  DeviceTensor<float, 2>& input,
  DeviceTensor<float, 2>& weight,
  DeviceTensor<float, 3>& output,
  bool featuresInDim2);

void launchLookupTableGPUAccGradParametersKernel(
  DeviceTensor<float, 2>& input,
  DeviceTensor<float, 3>& gradOutput,
  DeviceTensor<float, 2>& gradWeight,
  float scale,
  bool featuresInDim2);
}

namespace {

int updateOutput(lua_State* L) {
  const auto input  = (THCudaTensor*)luaT_checkudata(L, 1, "torch.CudaTensor");
  const auto weight = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  auto output = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
  bool featuresInDim2 = lua_toboolean(L, 4);

  if (!(THCudaTensor_isContiguous(input) &&
        THCudaTensor_isContiguous(weight) &&
        THCudaTensor_isContiguous(output))) {
    luaL_error(L, "Tensors must be contiguous");
  }

  const auto inputDims = THCudaTensor_nDimension(input);
  if (inputDims > 2) {
    luaL_error(L, "input tensor size must be 1 or 2 dims (2 for batch mode)");
  }

  if (THCudaTensor_nDimension(output) != inputDims + 1) {
    luaL_error(L, "input and output tensors must both be "
               "in batch mode or non-batch mode");
  }

  DeviceTensor<float, 2> cudaInput;
  DeviceTensor<float, 2> cudaWeight = torchToDeviceTensor<float, 2>(weight);
  DeviceTensor<float, 3> cudaOutput;

  if (inputDims == 1) {
    DeviceTensor<float, 1> input1d = torchToDeviceTensor<float, 1>(input);
    cudaInput = input1d.upcastOuter<2>();

    DeviceTensor<float, 2> output2d = torchToDeviceTensor<float, 2>(output);
    cudaOutput = output2d.upcastOuter<3>();

    featuresInDim2 = false;
  } else {
    cudaInput = torchToDeviceTensor<float, 2>(input);
    cudaOutput = torchToDeviceTensor<float, 3>(output);
  }

  detail::launchLookupTableGPUUpdateOutputKernel(
    cudaInput, cudaWeight, cudaOutput, featuresInDim2);

  return 0;
}

int accGradParameters(lua_State* L) {
  const auto input  = (THCudaTensor*)luaT_checkudata(L, 1, "torch.CudaTensor");
  const auto gradOutput =
    (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  auto gradWeight = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
  float scale = lua_tonumber(L, 4);
  bool featuresInDim2 = lua_toboolean(L, 5);

  if (!(THCudaTensor_isContiguous(input) &&
        THCudaTensor_isContiguous(gradOutput) &&
        THCudaTensor_isContiguous(gradWeight))) {
    luaL_error(L, "Tensors must be contiguous");
  }

  const auto inputDims = THCudaTensor_nDimension(input);
  if (inputDims > 2) {
    luaL_error(L, "input tensor size must be 1 or 2 dims (2 for batch mode)");
  }

  if (THCudaTensor_nDimension(gradOutput) != inputDims + 1) {
    luaL_error(L, "input and gradOutput tensors must both be "
               "in batch mode or non-batch mode");
  }

  DeviceTensor<float, 2> cudaInput;
  DeviceTensor<float, 3> cudaGradOutput;
  DeviceTensor<float, 2> cudaGradWeight =
    torchToDeviceTensor<float, 2>(gradWeight);

  if (inputDims == 1) {
    DeviceTensor<float, 1> input1d =
      torchToDeviceTensor<float, 1>(input);
    cudaInput = input1d.upcastOuter<2>();

    DeviceTensor<float, 2> gradOutput2d =
      torchToDeviceTensor<float, 2>(gradOutput);
    cudaGradOutput = gradOutput2d.upcastOuter<3>();

    featuresInDim2 = false;
  } else {
    cudaInput = torchToDeviceTensor<float, 2>(input);
    cudaGradOutput = torchToDeviceTensor<float, 3>(gradOutput);
  }

  detail::launchLookupTableGPUAccGradParametersKernel(
    cudaInput, cudaGradOutput, cudaGradWeight, scale, featuresInDim2);

  return 0;
}

const luaL_Reg functions[] = {
  {"LookupTableGPU_updateOutput", updateOutput},
  {"LookupTableGPU_accGradParameters", accGradParameters},
  {nullptr, nullptr},
};

} // namespace

void initLookupTableGPUCuda(lua_State* L) {
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, functions, "nn");
  lua_pop(L, 1);
}

}}}  // namespaces
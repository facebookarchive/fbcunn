/**
 * Copyright 2014 Facebook
 * @author Michael Mathieu (myrhev@fb.com)
 */

#include "cuda/DeviceTensor.cuh"
#include "Utils.h"
#include "DeviceTensorUtils.h"
#include "THC.h"

#include <lua.hpp>
#include <TH.h>
#include <luaT.h>

using namespace facebook::cuda;

namespace facebook { namespace deeplearning { namespace torch {

namespace detail {
void launchLookupTableGPUUpdateOutputKernel(
  cudaStream_t stream,
  DeviceTensor<float, 2>& input,
  DeviceTensor<float, 2>& weight,
  DeviceTensor<float, 3>& output);

void launchLookupTableGPUAccGradParametersKernel(
  cudaStream_t stream,
  DeviceTensor<float, 2>& input,
  DeviceTensor<float, 3>& gradOutput,
  DeviceTensor<float, 2>& gradWeight,
  float scale);
}

namespace {

int updateOutput(lua_State* L) {
  THCState* state = getCutorchState(L);
  const auto input  = (THCudaTensor*)luaT_checkudata(L, 1, "torch.CudaTensor");
  const auto weight = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  auto output = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
  bool featuresInDim2 = lua_toboolean(L, 4);

  THAssert(THCudaTensor_checkGPU(state, 3, input, weight, output));
  if (!(THCudaTensor_isContiguous(state, input) &&
        THCudaTensor_isContiguous(state, weight) &&
        THCudaTensor_isContiguous(state, output))) {
    luaL_error(L, "Tensors must be contiguous");
  }

  const auto inputDims = THCudaTensor_nDimension(state, input);
  if (inputDims > 2) {
    luaL_error(L, "input tensor size must be 1 or 2 dims (2 for batch mode)");
  }

  if (THCudaTensor_nDimension(state, output) != inputDims + 1) {
    luaL_error(L, "input and output tensors must both be "
               "in batch mode or non-batch mode");
  }

  DeviceTensor<float, 2> cudaInput;
  DeviceTensor<float, 2> cudaWeight =
    torchToDeviceTensor<float, 2>(state, weight);
  DeviceTensor<float, 3> cudaOutput;

  if (inputDims == 1) {
    // Feature dimension is already innermost
    DeviceTensor<float, 1> input1d =
      torchToDeviceTensor<float, 1>(state, input);
    cudaInput = input1d.upcastOuter<2>();

    DeviceTensor<float, 2> output2d =
      torchToDeviceTensor<float, 2>(state, output);
    cudaOutput = output2d.upcastOuter<3>();
  } else {
    cudaInput = torchToDeviceTensor<float, 2>(state, input);
    cudaOutput = torchToDeviceTensor<float, 3>(state, output);

    if (featuresInDim2) {
      // Put feature dimension as innermost
      cudaOutput = cudaOutput.transpose(1, 2);
    }
  }

  detail::launchLookupTableGPUUpdateOutputKernel(
    THCState_getCurrentStream(state),
    cudaInput, cudaWeight, cudaOutput);
  return 0;
}

int accGradParameters(lua_State* L) {
  THCState* state = getCutorchState(L);
  const auto input  = (THCudaTensor*)luaT_checkudata(L, 1, "torch.CudaTensor");
  const auto gradOutput =
    (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  auto gradWeight = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
  float scale = lua_tonumber(L, 4);
  bool featuresInDim2 = lua_toboolean(L, 5);

  THAssert(THCudaTensor_checkGPU(state, 3, input, gradOutput, gradWeight));

  if (!(THCudaTensor_isContiguous(state, input) &&
        THCudaTensor_isContiguous(state, gradOutput) &&
        THCudaTensor_isContiguous(state, gradWeight))) {
    luaL_error(L, "Tensors must be contiguous");
  }

  const auto inputDims = THCudaTensor_nDimension(state, input);
  if (inputDims > 2) {
    luaL_error(L, "input tensor size must be 1 or 2 dims (2 for batch mode)");
  }

  if (THCudaTensor_nDimension(state, gradOutput) != inputDims + 1) {
    luaL_error(L, "input and gradOutput tensors must both be "
               "in batch mode or non-batch mode");
  }

  DeviceTensor<float, 2> cudaInput;
  DeviceTensor<float, 3> cudaGradOutput;
  DeviceTensor<float, 2> cudaGradWeight =
    torchToDeviceTensor<float, 2>(state, gradWeight);

  if (inputDims == 1) {
    // Feature dimension is already innermost
    DeviceTensor<float, 1> input1d =
      torchToDeviceTensor<float, 1>(state, input);
    cudaInput = input1d.upcastOuter<2>();

    DeviceTensor<float, 2> gradOutput2d =
      torchToDeviceTensor<float, 2>(state, gradOutput);
    cudaGradOutput = gradOutput2d.upcastOuter<3>();
  } else {
    cudaInput = torchToDeviceTensor<float, 2>(state, input);
    cudaGradOutput = torchToDeviceTensor<float, 3>(state, gradOutput);

    if (featuresInDim2) {
      // Put feature dimension as innermost
      cudaGradOutput = cudaGradOutput.transpose(1, 2);
    }
  }

  detail::launchLookupTableGPUAccGradParametersKernel(
    THCState_getCurrentStream(state),
    cudaInput, cudaGradOutput, cudaGradWeight, scale);

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

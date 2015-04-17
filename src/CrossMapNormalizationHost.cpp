/**
 * Copyright 2014 Facebook
 * @author Tudor Bosman (tudorb@fb.com)
 */

#include "THC.h"
#include "CrossMapNormalization.cuh"
#include "Utils.h"
#include <luaT.h>
#include <lua.hpp>

namespace facebook { namespace deeplearning { namespace torch {

namespace {

// Forward pass
int updateOutput(lua_State* L) {
  THCState* state = getCutorchState(L);
  auto input = static_cast<THCudaTensor*>(
      luaT_checkudata(L, 2, "torch.CudaTensor"));
  auto output = static_cast<THCudaTensor*>(
      luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor"));


  int outputIdx = lua_gettop(L);
  auto squaredSum = static_cast<THCudaTensor*>(
      luaT_getfieldcheckudata(L, 1, "squaredSum", "torch.CudaTensor"));

  THAssert(THCudaTensor_checkGPU(state, 3, input, output, squaredSum));

  detail::CrossMapNormalizationParam param;
  param.kernelSize = luaT_getfieldcheckint(L, 1, "size");
  param.kernelRadius = param.kernelSize / 2;
  param.scale = luaT_getfieldchecknumber(L, 1, "scale");
  param.power = luaT_getfieldchecknumber(L, 1, "power");

  int ndims = THCudaTensor_nDimension(state, input);
  if (ndims != 3 && ndims != 4) {
    luaL_error(L, "Invalid input tensor dimension");
  }

  if (param.kernelSize % 2 == 0) {
    luaL_error(L, "Kernel size must be odd");
  }

  // Make tensors contiguous
  input  = THCudaTensor_newContiguous(state, input);
  output = THCudaTensor_newContiguous(state, output);

  // Resize derived tensors based on input
  THCudaTensor_resizeAs(state, output, input);
  THCudaTensor_resizeAs(state, squaredSum, input);

  param.batchSize = 1;
  int firstDim = 0;
  if (ndims == 4) {
    param.batchSize = THCudaTensor_size(state, input, 0);
    firstDim = 1;
  }

  param.numFeatures = THCudaTensor_size(state, input, firstDim);
  param.featureSize = THCudaTensor_stride(state, input, firstDim);

  detail::launchCrossMapNormalizationUpdateOutputKernel(
    THCState_getCurrentStream(state),
    THCudaTensor_data(state, input),
    THCudaTensor_data(state, output),
    THCudaTensor_data(state, squaredSum),
    param);
  lua_pushvalue(L, outputIdx);
  THCudaTensor_free(state, input);
  THCudaTensor_free(state, output);

  return 1;
}

// Backprop
int updateGradInput(lua_State* L) {
  THCState* state = getCutorchState(L);
  auto input = static_cast<THCudaTensor*>(
      luaT_checkudata(L, 2, "torch.CudaTensor"));
  auto gradOutput = static_cast<THCudaTensor*>(
      luaT_checkudata(L, 3, "torch.CudaTensor"));
  auto gradInput = static_cast<THCudaTensor*>(
      luaT_getfieldcheckudata(L, 1, "gradInput", "torch.CudaTensor"));
  int gradInputIdx = lua_gettop(L);
  auto squaredSum = static_cast<THCudaTensor*>(
      luaT_getfieldcheckudata(L, 1, "squaredSum", "torch.CudaTensor"));

  THAssert(THCudaTensor_checkGPU(state, 4, input,
                                 gradInput, gradOutput, squaredSum));

  detail::CrossMapNormalizationParam param;
  param.kernelSize = luaT_getfieldcheckint(L, 1, "size");
  param.kernelRadius = param.kernelSize / 2;
  param.scale = luaT_getfieldchecknumber(L, 1, "scale");
  param.power = luaT_getfieldchecknumber(L, 1, "power");

  int ndims = THCudaTensor_nDimension(state, input);
  if (ndims != 3 && ndims != 4) {
    luaL_error(L, "Invalid input tensor dimension");
  }

  if (param.kernelSize % 2 == 0) {
    luaL_error(L, "Kernel size must be odd");
  }

  // Make tensors contiguous
  input = THCudaTensor_newContiguous(state, input);
  gradOutput = THCudaTensor_newContiguous(state, gradOutput);
  gradInput = THCudaTensor_newContiguous(state, gradInput);

  // Resize derived tensors based on input
  THCudaTensor_resizeAs(state, gradInput, input);

  param.batchSize = 1;
  int firstDim = 0;
  if (ndims == 4) {
    param.batchSize = THCudaTensor_size(state, input, 0);
    firstDim = 1;
  }


  param.numFeatures = THCudaTensor_size(state, input, firstDim);
  param.featureSize = THCudaTensor_stride(state, input, firstDim);

  detail::launchCrossMapNormalizationUpdateGradInputKernel(
    THCState_getCurrentStream(state),
    THCudaTensor_data(state, input),
    THCudaTensor_data(state, gradOutput),
    THCudaTensor_data(state, squaredSum),
    THCudaTensor_data(state, gradInput),
    param);

  lua_pushvalue(L, gradInputIdx);
  THCudaTensor_free(state, gradOutput);
  THCudaTensor_free(state, gradInput);
  THCudaTensor_free(state, input);
  return 1;
}

const luaL_Reg functions[] = {
  {"CrossMapNormalization_updateOutput", updateOutput},
  {"CrossMapNormalization_updateGradInput", updateGradInput},
  {nullptr, nullptr},
};

}  // namespace

void initCrossMapNormalizationCuda(lua_State* L) {
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, functions, "nn");
  lua_pop(L, 1);
}

}}}  // namespaces

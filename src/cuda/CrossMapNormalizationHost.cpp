/**
 * Copyright 2014 Facebook
 * @author Tudor Bosman (tudorb@fb.com)
 */

#include "THC.h"
#include "CrossMapNormalization.cuh"
#include <luaT.h>
#include <lua.hpp>

namespace facebook { namespace deeplearning { namespace torch {

namespace {

// Forward pass
int updateOutput(lua_State* L) {
  auto input = static_cast<THCudaTensor*>(
      luaT_checkudata(L, 2, "torch.CudaTensor"));
  auto output = static_cast<THCudaTensor*>(
      luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor"));


  int outputIdx = lua_gettop(L);
  auto squaredSum = static_cast<THCudaTensor*>(
      luaT_getfieldcheckudata(L, 1, "squaredSum", "torch.CudaTensor"));

  detail::CrossMapNormalizationParam param;
  param.kernelSize = luaT_getfieldcheckint(L, 1, "size");
  param.kernelRadius = param.kernelSize / 2;
  param.scale = luaT_getfieldchecknumber(L, 1, "scale");
  param.power = luaT_getfieldchecknumber(L, 1, "power");

  int ndims = THCudaTensor_nDimension(NULL, input);
  if (ndims != 3 && ndims != 4) {
    luaL_error(L, "Invalid input tensor dimension");
  }

  if (param.kernelSize % 2 == 0) {
    luaL_error(L, "Kernel size must be odd");
  }

  // Make tensors contiguous
  input  = THCudaTensor_newContiguous(NULL, input);
  output = THCudaTensor_newContiguous(NULL, output);

  // Resize derived tensors based on input
  THCudaTensor_resizeAs(NULL, output, input);
  THCudaTensor_resizeAs(NULL, squaredSum, input);

  param.batchSize = 1;
  int firstDim = 0;
  if (ndims == 4) {
    param.batchSize = THCudaTensor_size(NULL, input, 0);
    firstDim = 1;
  }

  param.numFeatures = THCudaTensor_size(NULL, input, firstDim);
  param.featureSize = THCudaTensor_stride(NULL, input, firstDim);

  detail::launchCrossMapNormalizationUpdateOutputKernel(
      THCudaTensor_data(NULL, input),
      THCudaTensor_data(NULL, output),
      THCudaTensor_data(NULL, squaredSum),
      param);
  lua_pushvalue(L, outputIdx);
  THCudaTensor_free(NULL, input);
  THCudaTensor_free(NULL, output);

  return 1;
}

// Backprop
int updateGradInput(lua_State* L) {
  auto input = static_cast<THCudaTensor*>(
      luaT_checkudata(L, 2, "torch.CudaTensor"));
  auto gradOutput = static_cast<THCudaTensor*>(
      luaT_checkudata(L, 3, "torch.CudaTensor"));
  auto gradInput = static_cast<THCudaTensor*>(
      luaT_getfieldcheckudata(L, 1, "gradInput", "torch.CudaTensor"));
  int gradInputIdx = lua_gettop(L);
  auto squaredSum = static_cast<THCudaTensor*>(
      luaT_getfieldcheckudata(L, 1, "squaredSum", "torch.CudaTensor"));


  detail::CrossMapNormalizationParam param;
  param.kernelSize = luaT_getfieldcheckint(L, 1, "size");
  param.kernelRadius = param.kernelSize / 2;
  param.scale = luaT_getfieldchecknumber(L, 1, "scale");
  param.power = luaT_getfieldchecknumber(L, 1, "power");

  int ndims = THCudaTensor_nDimension(NULL, input);
  if (ndims != 3 && ndims != 4) {
    luaL_error(L, "Invalid input tensor dimension");
  }

  if (param.kernelSize % 2 == 0) {
    luaL_error(L, "Kernel size must be odd");
  }

  // Make tensors contiguous
  input = THCudaTensor_newContiguous(NULL, input);
  gradOutput = THCudaTensor_newContiguous(NULL, gradOutput);
  gradInput = THCudaTensor_newContiguous(NULL, gradInput);

  // Resize derived tensors based on input
  THCudaTensor_resizeAs(NULL, gradInput, input);

  param.batchSize = 1;
  int firstDim = 0;
  if (ndims == 4) {
    param.batchSize = THCudaTensor_size(NULL, input, 0);
    firstDim = 1;
  }


  param.numFeatures = THCudaTensor_size(NULL, input, firstDim);
  param.featureSize = THCudaTensor_stride(NULL, input, firstDim);

  detail::launchCrossMapNormalizationUpdateGradInputKernel(
      THCudaTensor_data(NULL, input),
      THCudaTensor_data(NULL, gradOutput),
      THCudaTensor_data(NULL, squaredSum),
      THCudaTensor_data(NULL, gradInput),
      param);

  lua_pushvalue(L, gradInputIdx);
  THCudaTensor_free(NULL, gradOutput);
  THCudaTensor_free(NULL, gradInput);
  THCudaTensor_free(NULL, input);
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


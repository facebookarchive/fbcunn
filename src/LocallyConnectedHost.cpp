/**
 * Copyright 2014 Facebook
 * @author Frank Jargstorff (fjargsto@fb.com)
 */

#include "THC.h"
#include "Utils.h"
#include "LocallyConnected.cuh"
#include <luaT.h>
#include <lua.hpp>

namespace facebook { namespace deeplearning { namespace torch {

namespace {

using namespace detail;

// Change a 3-d or 4-d tensor from standard Torch layout (P x H x W) or
// (B x P x H x W) to CUDA layout (H x W x P) or (B x H x W x P). If the
// forceContiguous flag is set (to true), the tensor is made contiguous.
//
void toCudaLayout(THCState* state, THCudaTensor* tensor) {
  int ndims = THCudaTensor_nDimension(state, tensor);
  if (ndims == 4) {                               // B x P x H x W
    THCudaTensor_transpose(state, tensor, tensor, 1, 3); // B x W x H x P
    THCudaTensor_transpose(state, tensor, tensor, 1, 2); // B x H x W x P
  } else {                                        // P x H x W
    THCudaTensor_transpose(state, tensor, tensor, 0, 2); // W x H x P
    THCudaTensor_transpose(state, tensor, tensor, 0, 1); // H x W x P
  }
}

// Change a 3-d or 4-d tensor from CUDA layout (H x W x P) or
// (B x H x W x P) to standard Torch layout (P x H x W) or (B x P x H x W).
//
void toStandardLayout(THCState* state, THCudaTensor* tensor) {
  int ndims = THCudaTensor_nDimension(state, tensor);
  if (ndims == 4) {                               // B x H x W x P
    THCudaTensor_transpose(state, tensor, tensor, 1, 2); // B x W x H x P
    THCudaTensor_transpose(state, tensor, tensor, 1, 3); // B x P x H x W
  } else {                                        // H x W x P
    THCudaTensor_transpose(state, tensor, tensor, 0, 1); // W x H x P
    THCudaTensor_transpose(state, tensor, tensor, 0, 2); // P x H x W
  }
}

void initializeParams(THCState* state,
                      LocallyConnectedParam* params,
                      const THCudaTensor* weight,
                      const THCudaTensor* input,
                      int dH = 1, int dW = 1) {
  // Notice: For CUDA version of the module, the weight tensor's
  // layout is H_o, W_o, H_k, W_k, P_o, P_i
  params->outputHeight =
    THCudaTensor_size(state, weight, kKernelOutputHeightDim);
  params->outputWidth  =
    THCudaTensor_size(state, weight, kKernelOutputWidthDim);
  params->kernelHeight =
    THCudaTensor_size(state, weight, kKernelHeightDim);
  params->kernelWidth  =
    THCudaTensor_size(state, weight, kKernelWidthDim);
  params->outputPlanes =
    THCudaTensor_size(state, weight, kKernelOutputPlaneDim);
  params->inputPlanes  =
    THCudaTensor_size(state, weight, kKernelPlaneDim);
  params->dH = dH;
  params->dW = dW;
  int ndims = THCudaTensor_nDimension(state, input);
  if (ndims == 4) {
    params->batchSize   = THCudaTensor_size(state, input, kBatchDim);
    params->inputHeight = THCudaTensor_size(state, input, kHeightDim);
    params->inputWidth  = THCudaTensor_size(state, input, kWidthDim);
  } else {
    params->batchSize   = 1;
    params->inputHeight = THCudaTensor_size(state, input, 0);
    params->inputWidth  = THCudaTensor_size(state, input, 1);
  }
}

void narrowTensors(THCState* state,
                   THCudaTensor* in, THCudaTensor* in1,
                   THCudaTensor* out, THCudaTensor* out1,
                   int index, int size) {
  THCudaTensor_narrow(state, in1, in, 0, index, size);
  THCudaTensor_narrow(state, out1, out, 0, index, size);
}

// Updates a cache in cuda layout.
//
// The input tensor is in standard Torch layout and the resulting
// cache in cuda layout. Depending on the actual transposition state
// tensor t is in, this may require making a copy or not. In any case
// the cache tensor c is always physically in cuda layout (i.e. contiguous).
//
// Parameters:
//    t - the source tensor. This tensor is expected in
//        standard Torch layout (batch x planes x height x width).
//    c - the cache tensor receiving t in cuda layout (batch x height x
//        width * planes).
//
void updateCache(THCState* state, THCudaTensor* t, THCudaTensor* c) {
  toCudaLayout(state, t);
  if (THCudaTensor_isContiguous(state, t)) {
    // if `t` is in correct layout, make `c` a view of `t`.
    THCudaTensor_set(state, c, t);
  } else {
    // if `t` isn't in correct layout it needs to be copied into
    // `c` in the correct format.
    THCudaTensor_resizeAs(state, c, t);
    THCudaTensor_copy(state, c, t);
  }
  // convert `t` back to standard output
  toStandardLayout(state, t);
}

// Forward pass
int updateOutput(lua_State* L) {
  THCState* state = getCutorchState(L);
  auto input = static_cast<THCudaTensor*>(
    luaT_checkudata(L, 2, "torch.CudaTensor"));
  auto output = static_cast<THCudaTensor*>(
    luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor"));
  // store output index
  int outputIdx = lua_gettop(L);

  auto weight = static_cast<THCudaTensor*>(
    luaT_getfieldcheckudata(L, 1, "weight", "torch.CudaTensor"));
  auto bias = static_cast<THCudaTensor*>(
    luaT_getfieldcheckudata(L, 1, "bias", "torch.CudaTensor"));

  // enforce precondition of no more than 256 input planes
  if (THCudaTensor_size(state, weight, 5) > 256) {
    luaL_error(L, "Cannot handle more than 256 input planes.");
  }

  auto inputCache = static_cast<THCudaTensor*>(
    luaT_getfieldcheckudata(L, 1, "input_cache", "torch.CudaTensor"));
  // The input tensor in Cuda layout must be contiguous.

  THAssert(THCudaTensor_checkGPU(state, 5, input, output,
                                 weight, bias, inputCache));

  updateCache(state, input, inputCache);

  int dW = luaT_getfieldchecknumber(L, 1, "dW");
  int dH = luaT_getfieldchecknumber(L, 1, "dH");

  LocallyConnectedParam params;
  initializeParams(state, &params, weight, inputCache, dH, dW);

  // Resize the output tensor to the CUDA layout with planes being fastest
  // varying dimension. Because all storage is overwritten by the kernel,
  // it is not necessary to tranpose and make contigous.
  if (THCudaTensor_nDimension(state, inputCache) == 4) {
    THCudaTensor_resize4d(state,
                          output, params.batchSize, params.outputHeight,
                          params.outputWidth, params.outputPlanes);
  } else {
    THCudaTensor_resize3d(state,
                          output, params.outputHeight, params.outputWidth,
                          params.outputPlanes);
  }
  locallyConnectedUpdateOutput(THCState_getCurrentStream(state),
                               THCudaTensor_data(state, inputCache),
                               THCudaTensor_data(state, weight),
                               THCudaTensor_data(state, bias),
                               THCudaTensor_data(state, output),
                               params);

  // The output tensor was produced in CUDA layout. Transpose back to standard
  // layout (without making contiguous).
  toStandardLayout(state, output);

  // push outputIdx onto the stack
  lua_pushvalue(L, outputIdx);

  return 1;
}

int updateGradInput(lua_State* L) {
  THCState* state = getCutorchState(L);
  auto gradInput = static_cast<THCudaTensor*>(
    luaT_getfieldcheckudata(L, 1, "gradInput", "torch.CudaTensor"));
  // store output index
  int gradInputIdx = lua_gettop(L);

  auto weight = static_cast<THCudaTensor*>(
    luaT_getfieldcheckudata(L, 1, "weight", "torch.CudaTensor"));

  int dW = luaT_getfieldchecknumber(L, 1, "dW");
  int dH = luaT_getfieldchecknumber(L, 1, "dH");

  // enforce precondition of no more than 256 input planes
  if (THCudaTensor_size(state, weight, 5) > 256) {
    luaL_error(L, "Cannot handle more than 32 input planes.");
  }

  auto inputCache = static_cast<THCudaTensor*>(
    luaT_getfieldcheckudata(L, 1, "input_cache", "torch.CudaTensor"));
  auto gradOutput = static_cast<THCudaTensor*>(
    luaT_checkudata(L, 3, "torch.CudaTensor"));
  auto gradOutputCache = static_cast<THCudaTensor*>(
    luaT_getfieldcheckudata(L, 1, "gradOutput_cache", "torch.CudaTensor"));

  bool gradOutCacheValid = luaT_getfieldcheckboolean(L, 1,
                                                     "gradOutputCacheIsValid");

  THAssert(THCudaTensor_checkGPU(state, 5, gradInput, weight, inputCache,
                                 gradOutput, gradOutputCache));

  updateCache(state, gradOutput, gradOutputCache);

  // Resize the result tensor (gradInput) to the CUDA layout with planes being
  // fastest varying dimension. Because all storage is overwritten by the
  // kernel, it is not necessary to tranpose and make contigous.
  THCudaTensor_resizeAs(state, gradInput, inputCache);
  LocallyConnectedParam params;
  initializeParams(state, &params, weight, inputCache, dH, dW);

  locallyConnectedUpdateGradInput(THCState_getCurrentStream(state),
                                  THCudaTensor_data(state, gradOutputCache),
                                  THCudaTensor_data(state, weight),
                                  THCudaTensor_data(state, gradInput),
                                  params);

  // convert gradInput to standard layout (without making contiguous).
  toStandardLayout(state, gradInput);

  // push outputIdx onto the stack
  lua_pushvalue(L, gradInputIdx);
  return 1;
}

int accGradParameters(lua_State* L) {
  THCState* state = getCutorchState(L);
  auto scale = static_cast<float>(
    luaL_checknumber(L, 4));

  auto gradWeight = static_cast<THCudaTensor*>(
    luaT_getfieldcheckudata(L, 1, "gradWeight", "torch.CudaTensor"));
  auto gradBias = static_cast<THCudaTensor*>(
    luaT_getfieldcheckudata(L, 1, "gradBias", "torch.CudaTensor"));

  int dW = luaT_getfieldchecknumber(L, 1, "dW");
  int dH = luaT_getfieldchecknumber(L, 1, "dH");

  auto inputCache = static_cast<THCudaTensor*>(
    luaT_getfieldcheckudata(L, 1, "input_cache", "torch.CudaTensor"));
  auto gradOutput = static_cast<THCudaTensor*>(
    luaT_checkudata(L, 3, "torch.CudaTensor"));
  auto gradOutputCache = static_cast<THCudaTensor*>(
    luaT_getfieldcheckudata(L, 1, "gradOutput_cache", "torch.CudaTensor"));
  bool gradOutCacheValid = luaT_getfieldcheckboolean(L, 1,

                                                     "gradOutputCacheIsValid");

  THAssert(THCudaTensor_checkGPU(state, 5, gradWeight, gradBias, inputCache,
                                 gradOutput, gradOutputCache));
  updateCache(state, gradOutput, gradOutputCache);

  LocallyConnectedParam params;
  initializeParams(state, &params, gradWeight, inputCache, dH, dW);

  locallyConnectedAccGradParameters(THCState_getCurrentStream(state),
                                    THCudaTensor_data(state, inputCache),
                                    THCudaTensor_data(state, gradOutputCache),
                                    THCudaTensor_data(state, gradWeight),
                                    THCudaTensor_data(state, gradBias),
                                    scale, params);

  return 0;
}

const luaL_Reg functions[] = {
  {"LocallyConnected_updateOutput", updateOutput},
  {"LocallyConnected_updateGradInput", updateGradInput},
  {"LocallyConnected_accGradParameters", accGradParameters},
  {nullptr, nullptr},
};

}  // anonymous namespace

void initLocallyConnectedCuda(lua_State* L) {
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, functions, "nn");
  lua_pop(L, 1);
}

}}}  // namespaces

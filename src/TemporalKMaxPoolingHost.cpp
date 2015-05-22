// Copyright 2004-present Facebook. All Rights Reserved.

#include "cuda/DeviceTensor.cuh"
#include "Utils.h"
#include "DeviceTensorUtils.h"
#include "THC.h"
#include "TemporalKMaxPooling.cuh"

#include <folly/ScopeGuard.h>
#include <lua.hpp>
#include <luaT.h>

using namespace facebook::cuda;

namespace facebook { namespace deeplearning { namespace torch {

namespace {

int checkAndAdjustK(lua_State* L, int k, double kDynamic, long sequenceLength) {
  if (kDynamic > 0) {
    k = std::max(k, (int) (kDynamic * sequenceLength));
  }

  if (k > sequenceLength) {
    luaL_error(L, "k (%d) must be less than sequence length (%d) ", k, sequenceLength);
  }

  return k;
}

int cunn_TemporalKMaxPooling_updateOutput(lua_State *L) {
  THCState* state = getCutorchState(L);
  auto inputTH = (THCudaTensor*) luaT_checkudata(L, 2, "torch.CudaTensor");
  auto indicesTH = (THCudaTensor*) luaT_getfieldcheckudata(
    L, 1, "indices", "torch.CudaTensor");
  auto outputTH = (THCudaTensor*) luaT_getfieldcheckudata(
    L, 1, "output", "torch.CudaTensor");

  THAssert(THCudaTensor_checkGPU(state, 3, inputTH, indicesTH, outputTH));

  auto k = luaT_getfieldcheckint(L, 1, "k");
  const auto kDynamic = luaT_getfieldchecknumber(L, 1, "k_dynamic");

  auto dimS = 0; // sequence dimension
  auto dimF = 1; // feature dimension

  luaL_argcheck(L, THCudaTensor_nDimension(state, inputTH) == 2 ||
                THCudaTensor_nDimension(state, inputTH) == 3, 2,
                "2D or 3D (batch mode) tensor expected");

  if (THCudaTensor_nDimension(state, inputTH) == 3) {
    dimS = 1;
    dimF = 2;
  }

  const auto sequenceLength = THCudaTensor_size(state, inputTH, dimS);
  const auto featureSize = THCudaTensor_size(state, inputTH, dimF);

  // get contiguous input
  auto inputContiguousTH = THCudaTensor_newContiguous(state, inputTH);
  SCOPE_EXIT{ THCudaTensor_free(state, inputContiguousTH); };

  k = checkAndAdjustK(L, k, kDynamic, sequenceLength);

  DeviceTensor<float, 3> input;
  DeviceTensor<float, 3> indices;
  DeviceTensor<float, 3> output;

  if (THCudaTensor_nDimension(state, inputContiguousTH) == 2) {
    // resize output
    THCudaTensor_resize2d(state, outputTH, k, featureSize);

    // indices will contain index locations for each output point
    THCudaTensor_resize2d(state, indicesTH, k, featureSize);

    input =
      torchToDeviceTensor<float, 2>(state, inputContiguousTH).upcastOuter<3>();
    output =
      torchToDeviceTensor<float, 2>(state, outputTH).upcastOuter<3>();
    indices =
      torchToDeviceTensor<float, 2>(state, indicesTH).upcastOuter<3>();
  } else {
    // number of batch frames
    const auto batchSize = THCudaTensor_size(state, inputContiguousTH, 0);

    // resize output
    THCudaTensor_resize3d(state, outputTH, batchSize, k, featureSize);

    // indices will contain index locations for each output point
    THCudaTensor_resize3d(state, indicesTH, batchSize, k, featureSize);

    input = torchToDeviceTensor<float, 3>(state, inputContiguousTH);
    indices = torchToDeviceTensor<float, 3>(state, indicesTH);
    output = torchToDeviceTensor<float, 3>(state, outputTH);
  }

  runTemporalKMaxPoolingUpdateOutput(
    THCState_getCurrentStream(state), input, indices, output, k);

  return 0;
}

int cunn_TemporalKMaxPooling_updateGradInput(lua_State *L) {
  THCState* state = getCutorchState(L);
  auto inputTH = (THCudaTensor*) luaT_checkudata(L, 2, "torch.CudaTensor");
  auto gradOutputTH = (THCudaTensor*) luaT_checkudata(L, 3, "torch.CudaTensor");
  auto indicesTH = (THCudaTensor*) luaT_getfieldcheckudata(
    L, 1, "indices", "torch.CudaTensor");
  auto gradInputTH = (THCudaTensor*) luaT_getfieldcheckudata(
    L, 1, "gradInput", "torch.CudaTensor");

  THAssert(THCudaTensor_checkGPU(state, 4, inputTH, indicesTH,
                                 gradInputTH, gradOutputTH));

  auto k = luaT_getfieldcheckint(L, 1, "k");
  const auto kDynamic = luaT_getfieldchecknumber(L, 1, "k_dynamic");

  // get contiguous gradOutput
  auto gradOutputContiguousTH =
    THCudaTensor_newContiguous(state, gradOutputTH);
  SCOPE_EXIT{ THCudaTensor_free(state, gradOutputContiguousTH); };

  // resize and zero
  THCudaTensor_resizeAs(state, gradInputTH, inputTH);
  THCudaTensor_zero(state, gradInputTH);

  DeviceTensor<float, 3> gradOutput;
  DeviceTensor<float, 3> indices;
  DeviceTensor<float, 3> gradInput;

  if (THCudaTensor_nDimension(state, inputTH) == 2) {
    gradOutput =
      torchToDeviceTensor<float, 2>(
        state, gradOutputContiguousTH).upcastOuter<3>();
    indices =
      torchToDeviceTensor<float, 2>(state, indicesTH).upcastOuter<3>();
    gradInput =
      torchToDeviceTensor<float, 2>(state, gradInputTH).upcastOuter<3>();
  } else {
    gradOutput = torchToDeviceTensor<float, 3>(state, gradOutputContiguousTH);
    indices = torchToDeviceTensor<float, 3>(state, indicesTH);
    gradInput = torchToDeviceTensor<float, 3>(state, gradInputTH);
  }

  const auto sequenceLength = gradInput.getSize(1);
  k = checkAndAdjustK(L, k, kDynamic, sequenceLength);

  runTemporalKMaxPoolingUpdateGradInput(
    THCState_getCurrentStream(state), gradOutput, indices, gradInput, k);

  return 0;
}

const luaL_Reg registry[] = {
  { "TemporalKMaxPooling_updateOutput",
    cunn_TemporalKMaxPooling_updateOutput },
  { "TemporalKMaxPooling_updateGradInput",
    cunn_TemporalKMaxPooling_updateGradInput },
  { nullptr, nullptr }
};

} // namespace

void initTemporalKMaxPoolingCuda(lua_State *L) {
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, registry, "nn");
  lua_pop(L, 1);
}

} } }

// Copyright 2004-present Facebook. All Rights Reserved.

#include "cuda/DeviceTensor.cuh"
#include "Utils.h"
#include "DeviceTensorUtils.h"
#include "THC.h"
#include "FeatureLPPooling.cuh"

#include <folly/Optional.h>
#include <folly/ScopeGuard.h>
#include <glog/logging.h>
#include <lua.hpp>
#include <luaT.h>

using namespace facebook::cuda;

namespace facebook { namespace deeplearning { namespace torch {

namespace {

constexpr int outputSize(int inputSize, int width, int stride) {
  return ((inputSize - width) / stride) + 1;
}

// non-batch mode:
// [feature dim]
// [feature dim][opt dim 1]
// [feature dim][opt dim 1][opt dim 2]
//
// batch mode:
// [batch dim][feature dim]
// [batch dim][feature dim][opt dim 1]
// [batch dim][feature dim][opt dim 1][opt dim 2]
folly::Optional<DeviceTensor<float, 4>>
  upcast(THCState* state, THCudaTensor* t, bool batchMode) {
  auto inputDim = THCudaTensor_nDimension(state, t);

  if (inputDim == 1) {
    if (batchMode) {
      return folly::none;
    } else {
      // [feature dim]
      return torchToDeviceTensor<float, 1>(state, t).
        upcastOuter<2>().upcastInner<4>();
    }
  } else if (inputDim == 2) {
    if (batchMode) {
      // [batch dim][feature dim]
      return torchToDeviceTensor<float, 2>(state, t).upcastInner<4>();
    } else {
      // [feature dim][opt dim 1]
      return torchToDeviceTensor<float, 2>(state, t).
        upcastOuter<3>().upcastInner<4>();
    }
  } else if (inputDim == 3) {
    if (batchMode) {
      // [batch dim][feature dim][opt dim 1]
      return torchToDeviceTensor<float, 3>(state, t).upcastInner<4>();
    } else {
      // [feature dim][opt dim 1][opt dim 2]
      return torchToDeviceTensor<float, 3>(state, t).upcastOuter<4>();
    }
  } else if (inputDim == 4) {
    if (batchMode) {
      // [batch dim][feature dim][opt dim 1][opt dim 2]
      return torchToDeviceTensor<float, 4>(state, t);
    } else {
      return folly::none;
    }
  }

  return folly::none;
}

// Resizes `toResize` based on the output size for `src` as an input
// tensor
void
resizeForOutput(THCState* state,
                THCudaTensor* toResize, THCudaTensor* input,
                bool batchMode, int width, int stride) {
  auto inputDim = THCudaTensor_nDimension(state, input);
  assert(inputDim >= 1 && inputDim <= 4);

  auto outSize = outputSize(THCudaTensor_size(state, input, 0), width, stride);
  if (batchMode) {
    assert(inputDim > 1);
    outSize = outputSize(THCudaTensor_size(state, input, 1), width, stride);
  } else {
    assert(inputDim < 4);
  }

  if (inputDim == 1) {
    THCudaTensor_resize1d(state, toResize, outSize);
  } else if (inputDim == 2) {
    if (batchMode) {
      THCudaTensor_resize2d(
        state, toResize, THCudaTensor_size(state, input, 0), outSize);
    } else {
      THCudaTensor_resize2d(
        state, toResize, outSize, THCudaTensor_size(state, input, 1));
    }
  } else if (inputDim == 3) {
    if (batchMode) {
      THCudaTensor_resize3d(
        state,
        toResize,
        THCudaTensor_size(state, input, 0), outSize,
        THCudaTensor_size(state, input, 2));
    } else {
      THCudaTensor_resize3d(
        state,
        toResize,
        outSize, THCudaTensor_size(state, input, 1),
        THCudaTensor_size(state, input, 2));
    }
  } else if (inputDim == 4) {
    THCudaTensor_resize4d(
      state,
      toResize,
      THCudaTensor_size(state, input, 0), outSize,
      THCudaTensor_size(state, input, 2), THCudaTensor_size(state, input, 3));
  }
}

// Makes `toResize` the same size/dimensionality as `src`
void
resize(THCState* state, THCudaTensor* toResize, THCudaTensor* src) {
  auto inputDim = THCudaTensor_nDimension(state, src);

  if (inputDim == 1) {
    THCudaTensor_resize1d(state, toResize,
                          THCudaTensor_size(state, src, 0));
  } else if (inputDim == 2) {
    THCudaTensor_resize2d(
      state,
      toResize, THCudaTensor_size(state, src, 0),
      THCudaTensor_size(state, src, 1));
  } else if (inputDim == 3) {
    THCudaTensor_resize3d(
      state,
      toResize,
      THCudaTensor_size(state, src, 0),
      THCudaTensor_size(state, src, 1),
      THCudaTensor_size(state, src, 2));
  } else if (inputDim == 4) {
    THCudaTensor_resize4d(
      state,
      toResize,
      THCudaTensor_size(state, src, 0), THCudaTensor_size(state, src, 1),
      THCudaTensor_size(state, src, 2), THCudaTensor_size(state, src, 3));
  } else {
    // should not encounter this dimensionality
    assert(false);
  }
}

int featureLPPooling_updateOutput(lua_State *L) {
  THCState* state = getCutorchState(L);
  auto inputTH = (THCudaTensor*) luaT_checkudata(L, 2, "torch.CudaTensor");
  auto outputTH = (THCudaTensor*) luaT_getfieldcheckudata(
    L, 1, "output", "torch.CudaTensor");

  auto width = luaT_getfieldcheckint(L, 1, "width");
  auto stride = luaT_getfieldcheckint(L, 1, "stride");
  auto power = luaT_getfieldchecknumber(L, 1, "power");
  auto batchMode = luaT_getfieldcheckboolean(L, 1, "batch_mode");

  THAssert(THCudaTensor_checkGPU(state, 2, inputTH, outputTH));

  DeviceTensor<float, 4> input;
  DeviceTensor<float, 4> output;

  auto inputUpcast = upcast(state, inputTH, batchMode);
  if (!inputUpcast) {
    if (batchMode) {
      luaL_error(L, "batch_mode: input must be 2-4 dimensions");
    } else {
      luaL_error(L, "no batch_mode: input must be 1-3 dimensions");
    }
  }
  input = *inputUpcast;

  // Make sure the feature dimension is properly sized
  if (input.getSize(1) < width) {
    luaL_error(L, "input: feature dimension must be >= width");
  }

  // Make sure that width and stride are within range
  if (width < 2 || width > 16) {
    luaL_error(L, "width: must be between 2 -> 16");
  }

  if (stride < 1 || stride > 4) {
    luaL_error(L, "stride: must be between 1 -> 4");
  }

  resizeForOutput(state, outputTH, inputTH, batchMode, width, stride);
  auto outputUpcast = upcast(state, outputTH, batchMode);
  assert(outputUpcast);
  output = *outputUpcast;

  bool found =
    runFeatureLPPoolingUpdateOutput(
      THCState_getCurrentStream(state), input, output, power, width, stride);
  assert(found);

  return 0;
}

int featureLPPooling_updateGradInput(lua_State *L) {
  THCState* state = getCutorchState(L);
  auto inputTH = (THCudaTensor*) luaT_checkudata(L, 2, "torch.CudaTensor");
  auto gradOutputTH = (THCudaTensor*) luaT_checkudata(L, 3, "torch.CudaTensor");

  auto outputTH = (THCudaTensor*) luaT_getfieldcheckudata(
    L, 1, "output", "torch.CudaTensor");
  auto gradInputTH = (THCudaTensor*) luaT_getfieldcheckudata(
    L, 1, "gradInput", "torch.CudaTensor");

  auto width = luaT_getfieldcheckint(L, 1, "width");
  auto stride = luaT_getfieldcheckint(L, 1, "stride");
  auto power = luaT_getfieldchecknumber(L, 1, "power");
  auto batchMode = luaT_getfieldcheckboolean(L, 1, "batch_mode");

  THAssert(THCudaTensor_checkGPU(state, 4, inputTH, outputTH,
                                 gradInputTH, gradOutputTH));

  DeviceTensor<float, 4> gradOutput;
  DeviceTensor<float, 4> input;
  DeviceTensor<float, 4> output;
  DeviceTensor<float, 4> gradInput;

  auto inputUpcast = upcast(state, inputTH, batchMode);
  if (!inputUpcast) {
    if (batchMode) {
      luaL_error(L, "batch_mode: input must be 2-4 dimensions");
    } else {
      luaL_error(L, "no batch_mode: input must be 1-3 dimensions");
    }
  }
  input = *inputUpcast;

  // Make sure the feature dimension is properly sized
  if (input.getSize(1) < width) {
    luaL_error(L, "input: feature dimension must be >= width");
  }

  // Make sure that width and stride are within range
  if (width < 2 || width > 16) {
    luaL_error(L, "width: must be between 2 -> 16");
  }

  if (stride < 1 || stride > 4) {
    luaL_error(L, "stride: must be between 1 -> 4");
  }

  auto gradOutputUpcast = upcast(state, gradOutputTH, batchMode);
  auto outputUpcast = upcast(state, outputTH, batchMode);

  if (!gradOutputUpcast || !outputUpcast) {
    luaL_error(L, "output and/or gradOutput are improperly sized");
  }

  gradOutput = *gradOutputUpcast;
  output = *outputUpcast;

  if (!output.isSameSizeAndStride(gradOutput)) {
    luaL_error(L, "output and gradOutput sizes do not match");
  }

  // Make sure that the input sizes produce the output sizes
  if (outputSize(input.getSize(1), width, stride) !=
      output.getSize(1)) {
    luaL_error(L, "input and output sizes do not match with respect to "
               "width and stride");
  }

  // Resize `gradInput` based on `input`
  resize(state, gradInputTH, inputTH);
  auto gradInputUpcast = upcast(state, gradInputTH, batchMode);
  assert(gradInputUpcast);

  gradInput = *gradInputUpcast;

  bool found =
    runFeatureLPPoolingUpdateGradInput(
      THCState_getCurrentStream(state),
      gradOutput, input, output, gradInput, power, width, stride);
  assert(found);

  return 0;
}

const luaL_Reg registry[] = {
  { "FeatureLPPooling_updateOutput",
    featureLPPooling_updateOutput },
  { "FeatureLPPooling_updateGradInput",
    featureLPPooling_updateGradInput },
  { nullptr, nullptr }
};

} // namespace

void initFeatureLPPoolingCuda(lua_State *L) {
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, registry, "nn");
  lua_pop(L, 1);
}

} } }

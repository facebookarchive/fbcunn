// Copyright 2004-present Facebook. All Rights Reserved.

#include "cuda/DeviceTensor.cuh"
#include "DeviceTensorUtils.h"
#include "util/Misc.h"
#include "Utils.h"
#include "lua.h"
#include "luaT.h"
#include "THC.h"

using namespace facebook::cuda;
using namespace facebook::deeplearning::torch;

namespace {

__device__ __forceinline__ int getBatch() {
  return blockIdx.x;
}

__device__ __forceinline__ int getOutputFrame() {
  return blockIdx.y;
}

__global__ void
temporalMaxPoolingUpdateOutput(DeviceTensor<float, 3> input,
                               DeviceTensor<float, 3> indices,
                               DeviceTensor<float, 3> output,
                               int kernelWidth,
                               int kernelStride) {
  for (int feature = threadIdx.x;
       feature < input.getSize(2); feature += blockDim.x) {
    int maxIndex = -1;
    float max = -FLT_MAX;

    for (int kernel = 0; kernel < kernelWidth; ++kernel) {
      int inputFrame = getOutputFrame() * kernelStride + kernel;
      float val = input[getBatch()][inputFrame][feature];

      if (val > max) {
        max = val;
        maxIndex = kernel;
      }
    }

    output[getBatch()][getOutputFrame()][feature] = max;
    indices[getBatch()][getOutputFrame()][feature] = maxIndex;
  }
}

__global__ void
temporalMaxPoolingUpdateGradInput(DeviceTensor<float, 3> gradOutput,
                                  DeviceTensor<float, 3> indices,
                                  DeviceTensor<float, 3> gradInput,
                                  int kernelStride) {
  int inputFrameBase = getOutputFrame() * kernelStride;

  for (int feature = threadIdx.x;
       feature < gradOutput.getSize(2); feature += blockDim.x) {
    int maxIndex = (int) indices[getBatch()][getOutputFrame()][feature];

    atomicAdd(
      &gradInput[getBatch()][inputFrameBase + maxIndex][feature],
      gradOutput[getBatch()][getOutputFrame()][feature]);
  }
}

} // namespace

static int fbcunn_TemporalMaxPooling_updateOutput(lua_State *L) {
  THCState* state = getCutorchState(L);
  THCudaTensor* inputTH =
    (THCudaTensor*) luaT_checkudata(L, 2, "torch.CudaTensor");
  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  THCudaTensor* indicesTH =
    (THCudaTensor*) luaT_getfieldcheckudata(
      L, 1, "indices", "torch.CudaTensor");
  THCudaTensor* outputTH =
    (THCudaTensor*) luaT_getfieldcheckudata(
      L, 1, "output", "torch.CudaTensor");

  THAssert(THCudaTensor_checkGPU(state, 3, inputTH, indicesTH, outputTH));

  int dimS = 0; // sequence dimension
  int dimF = 1; // feature dimension

  luaL_argcheck(L, THCudaTensor_nDimension(state, inputTH) == 2 ||
                THCudaTensor_nDimension(state, inputTH) == 3, 2,
                "2D or 3D(batch mode) tensor expected");

  if (THCudaTensor_nDimension(state, inputTH) == 3) {
    dimS = 1;
    dimF = 2;
  }

  luaL_argcheck(L, THCudaTensor_size(state, inputTH, dimS) >= kW, 2,
                "input sequence smaller than kernel size");

  // sizes
  long niframe = THCudaTensor_size(state, inputTH, dimS);
  long framesize = THCudaTensor_size(state, inputTH, dimF);
  long noframe = (niframe - kW) / dW + 1;

  // get contiguous input
  THCudaTensor* inputContiguousTH = THCudaTensor_newContiguous(state, inputTH);

  DeviceTensor<float, 3> input;
  DeviceTensor<float, 3> indices;
  DeviceTensor<float, 3> output;

  if (THCudaTensor_nDimension(state, inputContiguousTH) == 2) {
    // resize output
    THCudaTensor_resize2d(state, outputTH, noframe, framesize);

    // indices will contain index locations for each output point
    THCudaTensor_resize2d(state, indicesTH, noframe, framesize);

    input =
      torchToDeviceTensor<float, 2>(state, inputContiguousTH).upcastOuter<3>();
    output =
      torchToDeviceTensor<float, 2>(state, outputTH).upcastOuter<3>();
    indices =
      torchToDeviceTensor<float, 2>(state, indicesTH).upcastOuter<3>();
  } else {
    // number of batch frames
    long nbframe = THCudaTensor_size(state, inputContiguousTH, 0);

    // resize output
    THCudaTensor_resize3d(state, outputTH, nbframe, noframe, framesize);

    // indices will contain index locations for each output point
    THCudaTensor_resize3d(state, indicesTH, nbframe, noframe, framesize);

    input = torchToDeviceTensor<float, 3>(state, inputContiguousTH);
    indices = torchToDeviceTensor<float, 3>(state, indicesTH);
    output = torchToDeviceTensor<float, 3>(state, outputTH);
  }

  // Find the maximum number of threads that can fit onto our device
  // and that are less than the feature size. This kernel should not
  // be limited by smem or register count, so no need to use the
  // occupancy calculator.
  const cudaDeviceProp& deviceProperties =
    facebook::CUDAUtil::getCurrentDeviceProperties();

  dim3 block(min(input.getSize(2), deviceProperties.maxThreadsPerBlock));
  dim3 grid(input.getSize(0), // batch size
            output.getSize(1)); // # output frames

  temporalMaxPoolingUpdateOutput<<<grid, block,
    0, THCState_getCurrentStream(state)>>>(
    input, indices, output, kW, dW);

  // cleanup
  THCudaTensor_free(state, inputContiguousTH);

  return 1;
}

static int fbcunn_TemporalMaxPooling_updateGradInput(lua_State *L) {
  THCState* state = getCutorchState(L);
  THCudaTensor* inputTH =
    (THCudaTensor*) luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor* gradOutputTH =
    (THCudaTensor*) luaT_checkudata(L, 3, "torch.CudaTensor");
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  THCudaTensor* indicesTH =
    (THCudaTensor*) luaT_getfieldcheckudata(
      L, 1, "indices", "torch.CudaTensor");
  THCudaTensor* gradInputTH =
    (THCudaTensor*) luaT_getfieldcheckudata(
      L, 1, "gradInput", "torch.CudaTensor");

  THAssert(THCudaTensor_checkGPU(state, 4, inputTH, indicesTH,
                                 gradOutputTH, gradInputTH));

  // get contiguous gradOutput
  THCudaTensor* gradOutputContiguousTH =
    THCudaTensor_newContiguous(state, gradOutputTH);

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
    indices = torchToDeviceTensor<float, 2>(
      state, indicesTH).upcastOuter<3>();
    gradInput = torchToDeviceTensor<float, 2>(
      state, gradInputTH).upcastOuter<3>();
  } else {
    gradOutput = torchToDeviceTensor<float, 3>(state, gradOutputContiguousTH);
    indices = torchToDeviceTensor<float, 3>(state, indicesTH);
    gradInput = torchToDeviceTensor<float, 3>(state, gradInputTH);
  }

  // Find the maximum number of threads that can fit onto our device
  // and that are less than the feature size. This kernel should not
  // be limited by smem or register count, so no need to use the
  // occupancy calculator.
  const cudaDeviceProp& deviceProperties =
    facebook::CUDAUtil::getCurrentDeviceProperties();

  dim3 block(min(gradOutput.getSize(2), deviceProperties.maxThreadsPerBlock));
  dim3 grid(gradOutput.getSize(0), // batch size
            gradOutput.getSize(1)); // # output frames

  temporalMaxPoolingUpdateGradInput<<<grid, block,
    0, THCState_getCurrentStream(state)>>>(
    gradOutput, indices, gradInput, dW);

  // cleanup
  THCudaTensor_free(state, gradOutputContiguousTH);

  return 1;
}

static const struct luaL_Reg fbcunn_TemporalMaxPooling__[] = {
  { "TemporalMaxPooling_updateOutput",
    fbcunn_TemporalMaxPooling_updateOutput },
  { "TemporalMaxPooling_updateGradInput",
    fbcunn_TemporalMaxPooling_updateGradInput },
  { NULL, NULL }
};

static void fbcunn_TemporalMaxPooling_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  // Attach to nn instead of fbnn, since this is compatible with the
  // exsting nn.TemporalMaxPooling interface (unlike
  // fbnn.ClassNLLCriterion)
  luaT_registeratname(L, fbcunn_TemporalMaxPooling__, "nn");
  lua_pop(L, 1);
}

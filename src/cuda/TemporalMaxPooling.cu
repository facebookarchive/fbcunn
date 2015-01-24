// Copyright 2004-present Facebook. All Rights Reserved.

#include "cuda/DeviceTensor.cuh"
#include "DeviceTensorUtils.h"
#include "util/Misc.h"

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
      gradInput[getBatch()][inputFrameBase + maxIndex][feature].data(),
      gradOutput[getBatch()][getOutputFrame()][feature]);
  }
}

} // namespace

static int fbcunn_TemporalMaxPooling_updateOutput(lua_State *L) {
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

  int dimS = 0; // sequence dimension
  int dimF = 1; // feature dimension

  luaL_argcheck(L, THCudaTensor_nDimension(NULL, inputTH) == 2 ||
                THCudaTensor_nDimension(NULL, inputTH) == 3, 2,
                "2D or 3D(batch mode) tensor expected");

  if (THCudaTensor_nDimension(NULL, inputTH) == 3) {
    dimS = 1;
    dimF = 2;
  }

  luaL_argcheck(L, THCudaTensor_size(NULL, inputTH, dimS) >= kW, 2,
                "input sequence smaller than kernel size");

  // sizes
  long niframe = THCudaTensor_size(NULL, inputTH, dimS);
  long framesize = THCudaTensor_size(NULL, inputTH, dimF);
  long noframe = (niframe - kW) / dW + 1;

  // get contiguous input
  THCudaTensor* inputContiguousTH = THCudaTensor_newContiguous(NULL, inputTH);

  DeviceTensor<float, 3> input;
  DeviceTensor<float, 3> indices;
  DeviceTensor<float, 3> output;

  if (THCudaTensor_nDimension(NULL, inputContiguousTH) == 2) {
    // resize output
    THCudaTensor_resize2d(NULL, outputTH, noframe, framesize);

    // indices will contain index locations for each output point
    THCudaTensor_resize2d(NULL, indicesTH, noframe, framesize);

    input = torchToDeviceTensor<float, 2>(inputContiguousTH).upcastOuter<3>();
    output = torchToDeviceTensor<float, 2>(outputTH).upcastOuter<3>();
    indices = torchToDeviceTensor<float, 2>(indicesTH).upcastOuter<3>();
  } else {
    // number of batch frames
    long nbframe = THCudaTensor_size(NULL, inputContiguousTH, 0);

    // resize output
    THCudaTensor_resize3d(NULL, outputTH, nbframe, noframe, framesize);

    // indices will contain index locations for each output point
    THCudaTensor_resize3d(NULL, indicesTH, nbframe, noframe, framesize);

    input = torchToDeviceTensor<float, 3>(inputContiguousTH);
    indices = torchToDeviceTensor<float, 3>(indicesTH);
    output = torchToDeviceTensor<float, 3>(outputTH);
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

  temporalMaxPoolingUpdateOutput<<<grid, block>>>(
    input, indices, output, kW, dW);

  // cleanup
  THCudaTensor_free(NULL, inputContiguousTH);

  return 1;
}

static int fbcunn_TemporalMaxPooling_updateGradInput(lua_State *L) {
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

  // get contiguous gradOutput
  THCudaTensor* gradOutputContiguousTH =
    THCudaTensor_newContiguous(NULL, gradOutputTH);

  // resize and zero
  THCudaTensor_resizeAs(NULL, gradInputTH, inputTH);
  THCudaTensor_zero(NULL, gradInputTH);

  DeviceTensor<float, 3> gradOutput;
  DeviceTensor<float, 3> indices;
  DeviceTensor<float, 3> gradInput;

  if (THCudaTensor_nDimension(NULL, inputTH) == 2) {
    gradOutput =
      torchToDeviceTensor<float, 2>(gradOutputContiguousTH).upcastOuter<3>();
    indices = torchToDeviceTensor<float, 2>(indicesTH).upcastOuter<3>();
    gradInput = torchToDeviceTensor<float, 2>(gradInputTH).upcastOuter<3>();
  } else {
    gradOutput = torchToDeviceTensor<float, 3>(gradOutputContiguousTH);
    indices = torchToDeviceTensor<float, 3>(indicesTH);
    gradInput = torchToDeviceTensor<float, 3>(gradInputTH);
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

  temporalMaxPoolingUpdateGradInput<<<grid, block>>>(
    gradOutput, indices, gradInput, dW);

  // cleanup
  THCudaTensor_free(NULL, gradOutputContiguousTH);

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

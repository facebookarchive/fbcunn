// Copyright 2004-present Facebook. All Rights Reserved.

#include "ConvolutionBias.cuh"

#include "cuda/ComputeCapabilities.cuh"
#include "cuda/CudaUtils.cuh"
#include "cuda/DeviceTensor.cuh"
#include "cuda/WarpReductions.cuh"
#include "DeviceTensorUtils.h"
#include "util/Misc.h"

#include <boost/preprocessor/repetition/repeat.hpp>
#include <glog/logging.h>

using namespace facebook::cuda;
using namespace facebook::CUDAUtil;

// This layer computes the following:
//
// spatial bias:
//   output[batch][plane][row][col] += bias[plane]
//   (each GPU block is responsible for one plane)
//
// temporal bias:
//    output[batch][plane][time] += bias[time]
//    (each GPU block is responsible for one time)

namespace facebook { namespace deeplearning { namespace torch {
namespace bias {

__device__ __forceinline__ int spatialBiasGetBatch() {
  return blockIdx.y;
}

__device__ __forceinline__ int spatialBiasGetOutputPlane() {
  return blockIdx.x;
}

__global__ void updateOutputBias(DeviceTensor<float, 1> bias,
                                 DeviceTensor<float, 4> output) {
  const float perFilterBias = bias[spatialBiasGetOutputPlane()];

  for (int row = threadIdx.y; row < output.getSize(2); row += blockDim.y) {
    for (int col = threadIdx.x;
         col < output.getSize(3);
         col += blockDim.x) {
      output[spatialBiasGetBatch()][spatialBiasGetOutputPlane()][row][col] +=
        perFilterBias;
    }
  }
}

// Simple, baseline case looks good enough so far
__global__ void updateOutputTemporalBias(DeviceTensor<float, 1> bias,
                                         DeviceTensor<float, 3> output) {
  for (int x = threadIdx.x; x < output.getSize(2); x += blockDim.x) {
    output[spatialBiasGetBatch()][spatialBiasGetOutputPlane()][x] += bias[x];
  }
}

__global__
void accGradParametersBias(DeviceTensor<float, 1> gradBias,
                           DeviceTensor<float, 4> output,
                           float scale) {
  float sum = 0.0f;
  const unsigned int plane = spatialBiasGetOutputPlane();

  for (unsigned int idx = threadIdx.x;
       idx < output.getSize(0) * output.getSize(2) * output.getSize(3);
       idx += blockDim.x) {
    unsigned int batch = idx / (output.getSize(2) * output.getSize(3));
    unsigned int row   = (idx / output.getSize(3) ) % output.getSize(2);
    unsigned int col   = idx % output.getSize(3);
    sum += output[batch][plane][row][col];
  }
  // reduce within the warp
  sum = cuda::warpReduceSum(sum);

  // reduce within the block
  unsigned int warpIdx = threadIdx.x / WARP_SIZE;
  unsigned int nWarps  = ceil(blockDim.x, (unsigned int) WARP_SIZE);
  assert(nWarps < 32);
  __shared__ float perWarpSum[32];
  if (getLaneId() == 0) {
    perWarpSum[warpIdx] = sum;
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    sum = 0.0f;
    for (int i = 0; i < nWarps; ++i) {
      sum += perWarpSum[i];
    }

    gradBias[plane] = sum * scale;
  }
}

// Simple, baseline case looks good enough so far
__global__ void accGradParametersTemporalBias(DeviceTensor<float, 1> gradBias,
                                              DeviceTensor<float, 3> output,
                                              float scale) {

  float sum = 0.0f;
  const unsigned int time = blockIdx.x;

  for (unsigned int idx = threadIdx.x;
       idx < output.getSize(0) * output.getSize(1);
       idx += blockDim.x) {
    unsigned int batch = idx / output.getSize(1);
    unsigned int plane = idx % output.getSize(1);
    sum += output[batch][plane][time];
  }
  // reduce within the warp
  sum = cuda::warpReduceSum(sum);

  // reduce within the block
  unsigned int warpIdx = threadIdx.x / WARP_SIZE;
  unsigned int nWarps  = ceil(blockDim.x, (unsigned int) WARP_SIZE);
  assert(nWarps < 32);
  __shared__ float perWarpSum[32];
  if (getLaneId() == 0) {
    perWarpSum[warpIdx] = sum;
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    sum = 0.0f;
    for (int i = 0; i < nWarps; ++i) {
      sum += perWarpSum[i];
    }

    gradBias[time] = sum * scale;
  }
}

void
updateOutputBias(THCState* state,
                 THCudaTensor* outputTH,
                 THCudaTensor* biasTH) {
  DeviceTensor<float, 4> output =
    torchToDeviceTensor<float, 4>(state, outputTH);
  DeviceTensor<float, 1> bias =
    torchToDeviceTensor<float, 1>(state, biasTH);

  int device = 0;
  cudaError_t err = cudaGetDevice(&device);
  // Code must run with a GPU present
  CHECK_EQ(err, cudaSuccess);

  const cudaDeviceProp& deviceProperties = getDeviceProperties(device);

  // Code is written expecting the warp size to be WARP_SIZE,
  // so it can be used as a compile-time constant
  CHECK_EQ(WARP_SIZE, deviceProperties.warpSize);
  CHECK_EQ(bias.getSize(0), output.getSize(1));

  dim3 grid(output.getSize(1), output.getSize(0));
  dim3 block(deviceProperties.warpSize, 8); // 8 seems to be a good tradeoff

  updateOutputBias<<<grid, block,
    0, THCState_getCurrentStream(state)>>>(bias, output);
}

void
updateOutputTemporalBias(THCState* state,
                         THCudaTensor* outputTH,
                         THCudaTensor* biasTH) {
  DeviceTensor<float, 3> output =
    torchToDeviceTensor<float, 3>(state, outputTH);
  DeviceTensor<float, 1> bias =
    torchToDeviceTensor<float, 1>(state, biasTH);

  int device = 0;
  cudaError_t err = cudaGetDevice(&device);
  // Code must run with a GPU present
  CHECK_EQ(err, cudaSuccess);

  const cudaDeviceProp& deviceProperties = getDeviceProperties(device);

  CHECK_EQ(bias.getSize(0), output.getSize(2));

  dim3 grid(output.getSize(1), output.getSize(0));
  dim3 block(std::min(output.getSize(2), deviceProperties.maxThreadsPerBlock));

  updateOutputTemporalBias<<<grid, block,
    0, THCState_getCurrentStream(state)>>>(bias, output);
}


void
accGradParametersBias(THCState* state,
                      THCudaTensor* outputTH,
                      THCudaTensor* gradBiasTH,
                      float biasScale) {
  DeviceTensor<float, 4> output =
    torchToDeviceTensor<float, 4>(state, outputTH);
  DeviceTensor<float, 1> gradBias =
    torchToDeviceTensor<float, 1>(state, gradBiasTH);

  int device = 0;
  cudaError_t err = cudaGetDevice(&device);
  // Code must run with a GPU present
  CHECK_EQ(err, cudaSuccess);

  const cudaDeviceProp& deviceProperties = getDeviceProperties(device);

  // Code is written expecting the warp size to be WARP_SIZE,
  // so it can be used as a compile-time constant
  CHECK_EQ(WARP_SIZE, deviceProperties.warpSize);
  CHECK_EQ(gradBias.getSize(0), output.getSize(1));

  // each block gets one plane, to avoid inter-block atomic adds
  dim3 grid = dim3(output.getSize(1));

  const int kMaxBlockSize = 8 * WARP_SIZE;
  dim3 block = dim3(min(
    output.getSize(0) * output.getSize(2) * output.getSize(3),
    kMaxBlockSize));

  accGradParametersBias<<<grid, block,
    0, THCState_getCurrentStream(state)>>>(gradBias, output, biasScale);
}

void
accGradParametersTemporalBias(THCState* state,
                              THCudaTensor* outputTH,
                              THCudaTensor* gradBiasTH,
                              float biasScale) {
  DeviceTensor<float, 3> output =
    torchToDeviceTensor<float, 3>(state, outputTH);
  DeviceTensor<float, 1> gradBias =
    torchToDeviceTensor<float, 1>(state, gradBiasTH);
  CHECK_EQ(gradBias.getSize(0), output.getSize(2));

  int device = 0;
  cudaError_t err = cudaGetDevice(&device);
  // Code must run with a GPU present
  CHECK_EQ(err, cudaSuccess);

  const cudaDeviceProp& deviceProperties = getDeviceProperties(device);

  // each block gets one time, to avoid inter-block atomic adds
  dim3 grid = dim3(output.getSize(2));

  const int kMaxBlockSize = 8 * WARP_SIZE;
  dim3 block = dim3(min(
    output.getSize(0) * output.getSize(1),
    kMaxBlockSize));

  accGradParametersTemporalBias<<<grid, block,
    0, THCState_getCurrentStream(state)>>>(gradBias, output, biasScale);
}

} } } } // namespace

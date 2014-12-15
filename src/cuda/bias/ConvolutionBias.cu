// Copyright 2004-present Facebook. All Rights Reserved.

#include "bias/ConvolutionBias.cuh"

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
                           int batchTile,
                           float scale) {
  const int batchStart = spatialBiasGetBatch() * batchTile;
  const int batchEnd = min(output.getSize(0),
                           (spatialBiasGetBatch() + 1) * batchTile);
  const int plane = spatialBiasGetOutputPlane();

  float sum = 0.0f;

  for (int batch = batchStart; batch < batchEnd; ++batch) {
    for (int row = threadIdx.y; row < output.getSize(2); row += blockDim.y) {
      for (int col = threadIdx.x; col < output.getSize(3); col += blockDim.x) {
        sum += output[batch][plane][row][col];
      }
    }
  }

  // Reduce within the warp
  sum = cuda::warpReduceSum(sum);

  // Reduce within the block
  assert(blockDim.y <= 32);
  __shared__ float perWarpSum[32];
  if (getLaneId() == 0) {
    perWarpSum[threadIdx.y] = sum;
  }

  __syncthreads();

  if (getLaneId() == 0 && threadIdx.y == 0) {
    sum = 0.0f;
    for (int i = 0; i < blockDim.y; ++i) {
      sum += perWarpSum[i];
    }

    if (batchTile == output.getSize(0)) {
      // We are the only block to contribute to this plane
      gradBias[plane] = sum * scale;
    } else {
      // Multiple blocks are contributing to this plane
      atomicAdd(gradBias[plane].data(), sum * scale);
    }
  }
}

// Simple, baseline case looks good enough so far
__global__ void accGradParametersTemporalBias(DeviceTensor<float, 1> gradBias,
                                              DeviceTensor<float, 3> output,
                                              float scale) {
  for (int tx = threadIdx.x; tx < output.getSize(2); tx += blockDim.x) {
    float localSum = 0.0f;
    int by = blockIdx.y;
#define TEMPORAL_BIAS_UNROLL 8
    for ( ; by + TEMPORAL_BIAS_UNROLL * gridDim.y < output.getSize(0);
            by += TEMPORAL_BIAS_UNROLL * gridDim.y) {
#pragma unroll
      for (int u = 0; u < TEMPORAL_BIAS_UNROLL; ++u) {
        localSum += output[by + u * gridDim.y][blockIdx.x][tx];
      }
    }
#undef TEMPORAL_BIAS_UNROLL
    // Rest
    for (; by < output.getSize(0); by += gridDim.y) {
      localSum += output[by][blockIdx.x][tx];
    }
    atomicAdd(gradBias[tx].data(), localSum * scale);
  }
}

void
updateOutputBias(THCudaTensor* outputTH, THCudaTensor* biasTH) {
  DeviceTensor<float, 4> output = torchToDeviceTensor<float, 4>(outputTH);
  DeviceTensor<float, 1> bias = torchToDeviceTensor<float, 1>(biasTH);

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

  updateOutputBias<<<grid, block>>>(bias, output);
}

void
updateOutputTemporalBias(THCudaTensor* outputTH, THCudaTensor* biasTH) {
  DeviceTensor<float, 3> output = torchToDeviceTensor<float, 3>(outputTH);
  DeviceTensor<float, 1> bias = torchToDeviceTensor<float, 1>(biasTH);

  int device = 0;
  cudaError_t err = cudaGetDevice(&device);
  // Code must run with a GPU present
  CHECK_EQ(err, cudaSuccess);

  const cudaDeviceProp& deviceProperties = getDeviceProperties(device);

  CHECK_EQ(bias.getSize(0), output.getSize(2));

  dim3 grid(output.getSize(1), output.getSize(0));
  dim3 block(std::min(output.getSize(2), deviceProperties.maxThreadsPerBlock));

  updateOutputTemporalBias<<<grid, block>>>(bias, output);
}

void
accGradParametersBias(THCudaTensor* outputTH,
                      THCudaTensor* gradBiasTH,
                      float biasScale) {
  DeviceTensor<float, 4> output = torchToDeviceTensor<float, 4>(outputTH);
  DeviceTensor<float, 1> gradBias = torchToDeviceTensor<float, 1>(gradBiasTH);

  int device = 0;
  cudaError_t err = cudaGetDevice(&device);
  // Code must run with a GPU present
  CHECK_EQ(err, cudaSuccess);

  const cudaDeviceProp& deviceProperties = getDeviceProperties(device);

  // Code is written expecting the warp size to be WARP_SIZE,
  // so it can be used as a compile-time constant
  CHECK_EQ(WARP_SIZE, deviceProperties.warpSize);
  CHECK_EQ(gradBias.getSize(0), output.getSize(1));

  // 8 seems to be a maximum sweetspot for number of warps to run with
  // atomicAdd tiling, but there's no need to be larger than the
  // number of rows available.
  const int maxWarps = std::min(output.getSize(2), 8);

  CHECK_GE(deviceProperties.maxThreadsPerBlock /
           deviceProperties.warpSize, maxWarps);

  // 32 seems to be a sweet spot for atomicAdds. batchTileSize is the
  // number of minibatch elements each block will handle itself. Thus,
  // minibatch size / batchTileSize is the number of atomicAdds that
  // will be executed. In order to reach the sweet spot, the tile size
  // should be as small as possible such that minibatch size /
  // batchTileSize is <= 32. Just search powers of 2.
  int batchTileSize = 1;
  while (ceil(output.getSize(0), batchTileSize) > 32) {
    batchTileSize *= 2;
  }

  dim3 grid(output.getSize(1), ceil(output.getSize(0), batchTileSize));
  dim3 block(deviceProperties.warpSize, maxWarps);

  if (batchTileSize < output.getSize(0)) {
    // Because we're accumulating using atomicAdds, we're responsible
    // for wiping the data before adding into it.
    cudaMemsetAsync(gradBias.data(), 0, gradBias.getSize(0) * sizeof(float));
  }

  accGradParametersBias<<<grid, block>>>(
    gradBias, output, batchTileSize, biasScale);
}

void
accGradParametersTemporalBias(THCudaTensor* outputTH,
                              THCudaTensor* gradBiasTH,
                              float biasScale) {
  DeviceTensor<float, 3> output = torchToDeviceTensor<float, 3>(outputTH);
  DeviceTensor<float, 1> gradBias = torchToDeviceTensor<float, 1>(gradBiasTH);

  int device = 0;
  cudaError_t err = cudaGetDevice(&device);
  // Code must run with a GPU present
  CHECK_EQ(err, cudaSuccess);

  const cudaDeviceProp& deviceProperties = getDeviceProperties(device);

  dim3 grid;
  if (output.getSize(0) >= deviceProperties.multiProcessorCount) {
    grid = dim3(output.getSize(1), 1);
  } else {
    const int kTileSizeY =
      (output.getSize(1) + deviceProperties.multiProcessorCount - 1) /
      deviceProperties.multiProcessorCount;
    grid = dim3(output.getSize(1),
                (output.getSize(0) + kTileSizeY - 1) / kTileSizeY);
  }
  dim3 block(std::min(output.getSize(2), deviceProperties.maxThreadsPerBlock));

  accGradParametersTemporalBias<<<grid, block>>>(gradBias, output, biasScale);
}

} } } } // namespace

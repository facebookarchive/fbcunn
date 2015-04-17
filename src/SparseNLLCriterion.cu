/**
 * Copyright 2014 Facebook
 * @author Michael Mathieu (myrhev@fb.com)
 */

#include "cuda/CudaUtils.cuh"
#include "cuda/WarpReductions.cuh"
#include "util/Misc.h"

#include "SparseNLLCriterion.cuh"

using namespace facebook::cuda;

namespace facebook { namespace deeplearning { namespace torch {
namespace detail {

namespace {

// only one block.
// threadIdx.x splits K (ideally, is equal to K)
// threadIdx.y splits batchSize
__global__ void
updateOutput(const DeviceTensor<float, 2> targetIdx,
             const DeviceTensor<float, 2> targetP,
             const DeviceTensor<float, 2> input,
             DeviceTensor<float, 1> output,
             const int batchSize,
             const int K) {
  extern __shared__ float buffer[];

  // map (sum the correct input multiplied by the probabilities)
  float local_sum = 0.f;
  for (int i = threadIdx.y; i < batchSize; i += blockDim.y) {
    for (int j = threadIdx.x; j < K; j += blockDim.x) {
      local_sum += input[i][(int)(targetIdx[i][j] - 1)] * targetP[i][j];
    }
  }

  // reduce (sum all)
  local_sum = cuda::warpReduceSum(local_sum);
  if (cuda::getLaneId() == 0)
    buffer[cuda::getWarpId()] = local_sum;
  __syncthreads();
  if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
    local_sum = 0.f;
    for (int i = 0; i < cuda::ceil(blockDim.x * blockDim.y, 32u); ++i) {
      local_sum += buffer[i];
    }
    output[0] = -local_sum;
  }
}

// blockIdx.x * threadIdx.y splits batchSize
// threadIdx.x splits K (ideally is equal to K)
__global__ void
updateGradInput(const DeviceTensor<float, 2> targetIdx,
                const DeviceTensor<float, 2> targetP,
                DeviceTensor<float, 2> gradInput,
                int batchSize, int K) {
  const int batch_idx = blockIdx.x * blockDim.y + threadIdx.y;
  const int batch_dim = gridDim.x * blockDim.y;
  for (int i = batch_idx; i < batchSize; i += batch_dim) {
    for (int j = threadIdx.x; j < K; j += blockDim.x) {
      gradInput[i][(int)(targetIdx[i][j] - 0.5)] = - targetP[i][j];
    }
  }
}

} // namespace

void runSparseNLLCriterion_updateOutput(
  cudaStream_t stream,
  const DeviceTensor<float, 2>& targetIdx,
  const DeviceTensor<float, 2>& targetP,
  const DeviceTensor<float, 2>& input,
  DeviceTensor<float, 1>& output) {

  const cudaDeviceProp& deviceProperties =
    facebook::CUDAUtil::getCurrentDeviceProperties();
  const int maxThreads = deviceProperties.maxThreadsPerBlock;

  const int batchSize = targetP.getSize(0);
  const int K = targetP.getSize(1);
  dim3 blocks(1, 1, 1);
  int threadsx = min(K, maxThreads);
  dim3 threads(threadsx, max(1, maxThreads/threadsx), 1);
  size_t sharedSize = cuda::ceil(threads.x * threads.y * sizeof(float),
                                 (size_t)deviceProperties.warpSize);
  updateOutput<<<blocks, threads, sharedSize, stream>>>(
    targetIdx, targetP, input, output, batchSize, K);
}

void runSparseNLLCriterion_updateGradInput(
  cudaStream_t stream,
  const DeviceTensor<float, 2>& targetIdx,
  const DeviceTensor<float, 2>& targetP,
  DeviceTensor<float, 2>& gradInput) {

  const cudaDeviceProp& deviceProperties =
    facebook::CUDAUtil::getCurrentDeviceProperties();

  const int batchSize = targetP.getSize(0);
  const int K = targetP.getSize(1);
  const int nClasses = gradInput.getSize(1);
  cudaMemsetAsync(gradInput.data(), 0, nClasses * batchSize * sizeof(float), stream);
  int threadsx = min(K, deviceProperties.maxThreadsPerBlock);
  int threadsy = (threadsx > 128) ? 1 : (256 / threadsx);
  dim3 threads(threadsx, threadsy, 1);
  dim3 blocks(max(1, batchSize / threadsy), 1, 1);
  updateGradInput<<<blocks, threads, 0, stream>>>(
    targetIdx, targetP, gradInput, batchSize, K);
}

}}}} // namespaces

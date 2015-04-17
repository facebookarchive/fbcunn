// Copyright 2004-present Facebook. All Rights Reserved.

#include "cuda/DeviceTensor.cuh"
#include "cuda/TopKElements.cuh"
#include "cuda/DeviceTensor.cuh"
#include "util/Misc.h"
#include "THC.h"

using namespace facebook::cuda;

namespace facebook { namespace deeplearning { namespace torch {

namespace {

__device__ __forceinline__
int getUpdateOutputBatch(const DeviceTensor<float, 3>& input) {
  return blockIdx.y;
}

__device__ __forceinline__
int getUpdateOutputFeature(const DeviceTensor<float, 3>& input) {
  return blockIdx.x * blockDim.y + threadIdx.y;
}

// input: [batch][frame][embedding]
// output: [batch][K frames s.t. (f)(embedding) is the highest][embedding]
// ordered in original [frame] order

__global__ void
temporalKMaxPoolingUpdateOutput(DeviceTensor<float, 3> input,
                                DeviceTensor<float, 3> indices,
                                DeviceTensor<float, 3> output,
                                int k) {
  const int batch = getUpdateOutputBatch(input);
  const int feature = getUpdateOutputFeature(input);

  if (feature >= input.getSize(2)) {
    return;
  }

  DeviceTensor<float, 1> input1d(&input[batch][0][feature],
                                 (const int[1]){ input.getSize(1) },
                                 (const int[1]){ input.getSize(2) });
  DeviceTensor<float, 1> output1d(&output[batch][0][feature],
                                  (const int[1]){ k },
                                  (const int[1]){ output.getSize(2) });
  DeviceTensor<float, 1> indices1d(&indices[batch][0][feature],
                                   (const int[1]){ k },
                                   (const int[1]){ indices.getSize(2) });

  warpFindTopKElementsIndexOrder(input1d, output1d, indices1d, k);
}

__device__ __forceinline__
int getUpdateGradInputBatch() {
  return blockIdx.x;
}

__device__ __forceinline__
int getUpdateGradInputOutputFrame() {
  return blockIdx.y;
}

__global__ void
temporalKMaxPoolingUpdateGradInput(DeviceTensor<float, 3> gradOutput,
                                   DeviceTensor<float, 3> indices,
                                   DeviceTensor<float, 3> gradInput,
                                   int k) {
  const int batch = getUpdateGradInputBatch();
  const int outputFrame = getUpdateGradInputOutputFrame();

  for (int feature = threadIdx.x;
       feature < gradInput.getSize(2);
       feature += blockDim.x) {
    int index = (int) indices[batch][outputFrame][feature];

    atomicAdd(&gradInput[batch][index][feature],
              gradOutput[batch][outputFrame][feature]);
  }
}

}

void
runTemporalKMaxPoolingUpdateOutput(cudaStream_t stream,
                                   const DeviceTensor<float, 3>& input,
                                   const DeviceTensor<float, 3>& indices,
                                   DeviceTensor<float, 3>& output,
                                   int k) {
  const cudaDeviceProp& deviceProperties =
    facebook::CUDAUtil::getCurrentDeviceProperties();

  // We aim to run with 4 warps.
  const int numWarps = std::min(input.getSize(2), 4);

  dim3 block(deviceProperties.warpSize, numWarps);
  dim3 grid(cuda::ceil(input.getSize(2), numWarps), input.getSize(0));

  temporalKMaxPoolingUpdateOutput<<<grid, block, 0, stream>>>(
    input, indices, output, k);
}

void
runTemporalKMaxPoolingUpdateGradInput(cudaStream_t stream,
                                      const DeviceTensor<float, 3>& gradOutput,
                                      const DeviceTensor<float, 3>& indices,
                                      DeviceTensor<float, 3>& gradInput,
                                      int k) {
  const cudaDeviceProp& deviceProperties =
    facebook::CUDAUtil::getCurrentDeviceProperties();

  // We aim to run with 4 warps.
  const int numThreads =
    std::min(gradOutput.getSize(2), deviceProperties.warpSize * 4);

  dim3 block(numThreads);
  dim3 grid(gradOutput.getSize(0),
            gradOutput.getSize(1));

  temporalKMaxPoolingUpdateGradInput<<<grid, block, 0, stream>>>(
    gradOutput, indices, gradInput, k);
}

} } }

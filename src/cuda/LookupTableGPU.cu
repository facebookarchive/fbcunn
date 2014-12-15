/**
 * Copyright 2014 Facebook
 * @author Michael Mathieu (myrhev@fb.com)
 */

#include "cuda/DeviceTensor.cuh"

using namespace facebook::cuda;

namespace facebook { namespace deeplearning { namespace torch {
namespace detail {

namespace {

__device__ __forceinline__ int getBatch() {
  return blockIdx.x;
}

__device__ __forceinline__ int getLookupElement() {
  return blockIdx.y;
}


template<bool F2>
__global__ void updateOutputKernel(DeviceTensor<float, 2> input,
                                     DeviceTensor<float, 2> weight,
                                     DeviceTensor<float, 3> output) {
  int weightIndex = (int)(input[getBatch()][getLookupElement()] - 0.5f);

  for (int i = threadIdx.x; i < weight.getSize(1); i += blockDim.x) {
    float val = weight[weightIndex][i];
    if (F2) {
      output[getBatch()][i][getLookupElement()] = val;
    } else {
      output[getBatch()][getLookupElement()][i] = val;
    }
  }
}

template<bool F2>
__global__ void accGradParametersKernel(DeviceTensor<float, 2> input,
                                        DeviceTensor<float, 3> gradOutput,
                                        DeviceTensor<float, 2> gradWeight,
                                        float scale) {
  int weightIndex = (int)(input[getBatch()][getLookupElement()] - 0.5f);
  for (int i = threadIdx.x; i < gradWeight.getSize(1); i += blockDim.x) {
    float src = F2 ? gradOutput[getBatch()][i][getLookupElement()] :
                     gradOutput[getBatch()][getLookupElement()][i];
    atomicAdd(gradWeight[weightIndex][i].data(), src);
  }
}

} // namespace

typedef DeviceTensor<float, 2> DeviceTensor2;
typedef DeviceTensor<float, 3> DeviceTensor3;

void launchLookupTableGPUUpdateOutputKernel(DeviceTensor2& input,
                                            DeviceTensor2& weight,
                                            DeviceTensor3& output,
                                            bool featuresInDim2) {
  const dim3 block(min(weight.getSize(1), 1024));
  const dim3 grid(input.getSize(0), input.getSize(1));

  if (featuresInDim2) {
    updateOutputKernel<true><<<grid, block>>>(input, weight, output);
  } else {
    updateOutputKernel<false><<<grid, block>>>(input, weight, output);
  }
}

void launchLookupTableGPUAccGradParametersKernel(DeviceTensor2& input,
                                                 DeviceTensor3& gradOutput,
                                                 DeviceTensor2& gradWeight,
                                                 float scale,
                                                 bool featuresInDim2) {
  const dim3 block(min(gradWeight.getSize(1), 1024));
  const dim3 grid(input.getSize(0), input.getSize(1));

  if (featuresInDim2) {
    accGradParametersKernel<true><<<grid, block>>>(
      input, gradOutput, gradWeight, scale);
  } else {
    accGradParametersKernel<false><<<grid, block>>>(
      input, gradOutput, gradWeight, scale);
  }
}


}}}}  // namespaces

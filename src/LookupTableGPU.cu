// Copyright 2004-present Facebook. All Rights Reserved.

#include "cuda/CudaUtils.cuh"
#include "cuda/DeviceTensor.cuh"
#include "cuda/WarpReductions.cuh"

using namespace facebook::cuda;

namespace facebook { namespace deeplearning { namespace torch {
namespace detail {

namespace {

// for updateOutput

__device__ __forceinline__ int getBatch() {
  return blockIdx.x;
}

__device__ __forceinline__ int getLookupElement() {
  return blockIdx.y;
}

// // for accGradParameters

__device__ __forceinline__ int getFeatureDim() {
  // Each warp runs effectively independently, but there is slightly
  // better utilization if each block has at least 4 warps.
  int warpId = threadIdx.x / 32;
  return blockIdx.x * 4 + warpId;
}

// Feature dimension is always innermost. Depending on tensor layout,
// it may or may not be contiguous.
__global__ void updateOutputKernel(DeviceTensor<float, 2> input,
                                   DeviceTensor<float, 2> weight,
                                   DeviceTensor<float, 3> output) {
  int weightIndex = (int)(input[getBatch()][getLookupElement()] - 0.5f);

  for (int i = threadIdx.x; i < weight.getSize(1); i += blockDim.x) {
    output[getBatch()][getLookupElement()][i] = weight[weightIndex][i];
  }
}

__global__ void accGradParametersKernel(DeviceTensor<float, 2> input,
                                        DeviceTensor<float, 3> gradOutput,
                                        DeviceTensor<float, 2> gradWeight,
                                        float scale) {
  const int featureDim = getFeatureDim();
  if (featureDim >= gradWeight.getSize(1)) {
    return;
  }

  // The strategy here is that each warp handles a single feature
  // dimension.
  // Within that feature dimension, points in the [batch][element]
  // dimension can overlap, and we need to determine if threads want
  // to add to the gradient in a colliding manner.
  // Typically one would use floating-point atomicAdd() to resolve
  // these collisions, but that is non-deterministic if there are
  // collisions. Non-determinism for this code is really bad,
  // especially in RNNs, and is prone to snowballing error.
  // In order to get a deterministic order of execution, we handle
  // non-colliding updates separately from colliding ones. Colliding
  // updates are serialized in their order of execution by using the
  // warp-wide collision detector `warpHasCollision`.
  unsigned int maxLinearIndex = input.getSize(0) * input.getSize(1);
  for (unsigned int i = getLaneId(); i < maxLinearIndex; i += WARP_SIZE) {
    unsigned int batch = i / input.getSize(1);
    unsigned int lookupElement = i % input.getSize(1);

    int weightIndex = (int) (input[batch][lookupElement].ldg() - 0.5f);
    float update = gradOutput[batch][lookupElement][featureDim] * scale;

    // Check for collision
    if (warpHasCollision(weightIndex)) {
      // Run all lanes sequentially; warp divergence
      for (int i = 0; i < WARP_SIZE; ++i) {
        if (getLaneId() == i) {
          gradWeight[weightIndex][featureDim] += update;
        }
      }
    } else {
      // No collision; warp coherence
      gradWeight[weightIndex][featureDim] += update;
    }
  }
}

} // namespace

typedef DeviceTensor<float, 2> DeviceTensor2;
typedef DeviceTensor<float, 3> DeviceTensor3;

void launchLookupTableGPUUpdateOutputKernel(cudaStream_t stream,
                                            DeviceTensor2& input,
                                            DeviceTensor2& weight,
                                            DeviceTensor3& output) {
  const dim3 grid(input.getSize(0), input.getSize(1));
  const dim3 block(min(weight.getSize(1), 1024));

  updateOutputKernel<<<grid, block, 0, stream>>>(input, weight, output);
}

void launchLookupTableGPUAccGradParametersKernel(cudaStream_t stream,
                                                 DeviceTensor2& input,
                                                 DeviceTensor3& gradOutput,
                                                 DeviceTensor2& gradWeight,
                                                 float scale) {
  // Target 4 warps/block for better utilization. Even if the input
  // doesn't have that many dimensions, the blocks/warps not
  // participating will just exit immediately.
  const dim3 grid(ceil(gradOutput.getSize(2), 4));
  const dim3 block(32 * 4);

  accGradParametersKernel<<<grid, block, 0, stream>>>(
    input, gradOutput, gradWeight, scale);
}

}}}}  // namespaces

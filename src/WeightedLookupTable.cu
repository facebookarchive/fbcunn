/**
 * Copyright 2015 Facebook
 */

#include "cuda/CudaUtils.cuh"
#include "cuda/DeviceTensor.cuh"
#include "cuda/WarpReductions.cuh"

using namespace facebook::cuda;

namespace facebook { namespace deeplearning { namespace torch {
namespace detail {

namespace {

__global__ void scaleByWeight(DeviceTensor<float, 2> output,
                              DeviceTensor<float, 2> input,
                              DeviceTensor<float, 1> weights) {
  // Values computed per thread
  const int VT = 4;

  // Each block computes a 4x128 section of the output, with each
  // warp handling a 1x128 section.

  int rowIdx = blockIdx.x * blockDim.y + threadIdx.y;
  if (rowIdx < weights.getSize(0)) {
    float weight = weights[rowIdx];

    #pragma unroll
    for (int i = 0; i < VT; i++) {
      int colIdx = blockDim.x * (VT * blockIdx.y + i) + threadIdx.x;
      if (colIdx < input.getSize(1)) {
        output[rowIdx][colIdx] = input[rowIdx][colIdx] * weight;
      }
    }
  }
}

}

void launchWeightedLookupTableScaleByWeightKernel(cudaStream_t stream,
                                                  DeviceTensor<float, 2>& output,
                                                  DeviceTensor<float, 2>& input,
                                                  DeviceTensor<float, 1>& weight) {
  dim3 grid(cuda::ceil(output.getSize(0), 4), cuda::ceil(output.getSize(1), 128));
  dim3 block(32, 4);

  scaleByWeight<<<grid, block, 0, stream>>>(output, input, weight);
}

}}}}

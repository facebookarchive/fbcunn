// Copyright 2004-present Facebook. All Rights Reserved.

#include "torch/fb/fbcunn/test/InputCentricConvolution_UpdateOutput.cuh"
#include "cuda/CudaUtils.cuh"
#include "cuda/DeviceTensor.cuh"
#include "torch/fb/fbcunn/src/DeviceTensorUtils.h"
#include "torch/fb/fbcunn/src/util/Misc.h"

#include <glog/logging.h>
#include <iostream>

using namespace facebook::cuda;
using namespace facebook::CUDAUtil;

namespace facebook { namespace deeplearning { namespace torch { namespace test {

// Given FilterSize registers and FilterStride step:
//  1. in-register shift of the FilterSize - FilterStride first registers
//  2. memory fetch the FilterStride last registers
template <int FilterSize, int FilterStride>
__device__ __forceinline__
void rotateAndFetch(float r[FilterSize], float* base, int stride) {
  for (int i = 0; i < FilterSize - FilterStride; ++i) {
    r[i] = r[i + FilterStride];
  }
  for (int i = FilterSize - FilterStride; i < FilterSize; ++i) {
    r[i]  = __ldg(base + (i - (FilterSize - FilterStride)) * stride);
  }
}

// Compile time tradeoff between threads per block and min blocks per SM
// influences the register + spilling usage. It seems we can afford a little
// spilling to get better utilization.
#define ICC_MAX_THREADS_PER_BLOCK 96
#define ICC_MIN_BLOCKS_PER_MULTIPROCESSOR 4
template <int FilterRowSize,
          int FilterColSize,
          int FilterRowStride,
          int FilterColStride,
          int NumInputPlanes,
          int NumInputCols>
__global__ void
__launch_bounds__(ICC_MAX_THREADS_PER_BLOCK, ICC_MIN_BLOCKS_PER_MULTIPROCESSOR)
InputCentricRelayoutConvolution_UpdateOutputDevice(
  DeviceTensor<float, 4> input,
  DeviceTensor<float, 4> filters,
  DeviceTensor<float, 4> output,
  int tileRow,
  int inputColMin,
  int inputColMax,
  int filterMin,
  int filterMax,
  int planeMin,
  int tileBatch) {

  // Failsafe tile along filters, generally, make sure it is an exact multiple
  // otherwise nasty thread divergence will happen. Especially on the atomic
  // adds to global memory.
  int filter = filterMin + threadIdx.x;
  if (filter >= filterMax) { return; }

  // Failsafe tile along batch sizes
  int batchMin = blockIdx.x * tileBatch;
  int batchMax = min(blockIdx.x * tileBatch + tileBatch,
                     (unsigned int) input.getSize(0));
  if (batchMin >= batchMax) { return; }

  // Cyclic vs block-cyclic reduce contention in the global memory load
  // Each threadIdx.y warp will contribute to 'ceil' consecutive output rows.
  // The cyclic approach scatters the contributions of each threadIdx.y and
  // reduces contention.
  int inputRows = input.getSize(2);
  int inputRow = cuda::ceil(inputRows, tileRow) * threadIdx.y + blockIdx.y;
  // This exhibits higher contention, all threadIdx.y hammer the same 'ceil'
  //   output rows at the same time.
  // int inputRow = blockIdx.y * tileRow + threadIdx.y;
  if (inputRow >= input.getSize(2)) { return; }

  // batch x row x col x filter layput needed for this kernel
  const int outputRows = output.getSize(1);
  const int outputCols =
    (NumInputCols - FilterColSize) / FilterColStride + 1;

  // Reference filterRow (i.e. the min row in the filter) that this point
  // convolves with. The other rows are filterRow0 + k * FilterRowStride
  const int filterRow0 = inputRow % FilterRowStride;
  // nvcc produces better code if we don't use the static FilterRowSize below
  const int ubRow = min(filters.getSize(1) - 1, inputRow);
  const int outputRow0 = (inputRow - filterRow0) / FilterRowStride;

  // Q: How many output rows does a single input point contribute to ?
  // A: At most ceil(FilterRowSize / FilterRowStride) for inner points, fewer
  //    on the boundary.
  // Therefore, pull in a local set of registers all the filter points needed
  // for FilterRowSize x ceil x NumInputPlanes. To keep everything compile
  // time static, fill the rest with zero (for boundary points).
  // The zeros will contribute flops that produce zero.
  // This is useless work but it vastly outperforms messing up the tight FMA
  // loop with control-flow adjustements.
  const int ceilFilterSizeFilterStride =
    (FilterRowSize + FilterRowStride - 1) / FilterRowStride;
  float filt[FilterColSize][NumInputPlanes][ceilFilterSizeFilterStride];
  for (int fx = 0; fx < ceilFilterSizeFilterStride; fx++) {
    int row = filterRow0 + fx * FilterRowStride;
    if (row <= ubRow) {
      for (int filterColIt = 0; filterColIt < FilterColSize; ++filterColIt) {
        for (int plane = 0; plane < NumInputPlanes; ++plane) {
          filt[filterColIt][plane][fx] =
            filters
            [planeMin + plane]
            [row]                    // depends on threadIdx.y
            [filterColIt]
            [filter].ldg();      // depends on threadIdx.x
        }
      }
    } else {
      for (int filterColIt = 0; filterColIt < FilterColSize; ++filterColIt) {
        for (int plane = 0; plane < NumInputPlanes; ++plane) {
          filt[filterColIt][plane][fx] = 0.0f;
        }
      }
    }
  }

// The BODY below is peeled once out of the main loop to avoid a conditional
// around the rotateAndFetch instruction
#define BODY(INDUCTION_VAR)                                             \
  float vals[ceilFilterSizeFilterStride];                               \
  for (int fx = 0; fx < ceilFilterSizeFilterStride; fx++) {             \
    vals[fx] = 0.0f;                                                    \
    for (int filterColIt = 0; filterColIt < FilterColSize; ++filterColIt) { \
      for (int plane = 0; plane < NumInputPlanes; ++plane) {            \
        vals[fx] += in[plane][filterColIt] * filt[filterColIt][plane][fx]; \
      }                                                                 \
    }                                                                   \
  }                                                                     \
  for (int fx = ceilFilterSizeFilterStride - 1; fx >= 0; --fx) {        \
    const int outputRow = outputRow0 - fx;                              \
    /* Padding output and writing out of bounds gives a nice 10% boost */ \
    /* if (outputRow < 0 || outputRow >= outputRows) { continue; } */   \
    float *o = &output[batch]                                           \
      [outputRow + ceilFilterSizeFilterStride]                          \
      [INDUCTION_VAR];                                                  \
    atomicAdd(o + filter, vals[fx]);                                    \
  }

  for (int batch = batchMin; batch < batchMax; ++batch) {
    // Load first set of FilterColSize points, then rotate by FilterColStride
    // at every outputCol iteration
    float in[NumInputPlanes][FilterColSize];
    for (int plane = 0; plane < NumInputPlanes; ++plane) {
      for (int filterColIt = 0; filterColIt < FilterColSize; ++filterColIt) {
        in[plane][filterColIt] =
          input[batch][planeMin + plane][inputRow][filterColIt].ldg();
      }
    }
    for (int outputCol = 0; outputCol < outputCols - 1; ++outputCol) {
      BODY(outputCol);
      // Load next set of FilterColSize points, this rotates by
      // FilterColStride at every iteration
      for (int plane = 0; plane < NumInputPlanes; ++plane) {
        int indCol = (outputCol + 1) * FilterColStride +
          (FilterColSize - FilterColStride);
        rotateAndFetch<FilterColSize, FilterColStride>
          (in[plane],
           &input[batch][planeMin + plane][inputRow][indCol],
           input.getStride(3));
      }
    }
    // Peeled last iteration to avoid inner conditional around rotate and fetch
    BODY(outputCols - 1);
  }

#undef BODY
}



template <int FilterRowSize,
          int FilterColSize,
          int FilterRowStride,
          int FilterColStride,
          int NumInputPlanes,
          int NumInputCols>
bool InputCentricRelayoutConvolution_UpdateOutputHost(
    const DeviceTensor<float, 4>& input,
    const DeviceTensor<float, 4>& filters,
    DeviceTensor<float, 4>& output) {

  int device = 0;
  cudaError_t err = cudaGetDevice(&device);
  // Code must run with a GPU present
  CHECK_EQ(err, cudaSuccess);

  const cudaDeviceProp& deviceProperties = getDeviceProperties(device);

  // Set max cache size
#define KERNEL \
  InputCentricRelayoutConvolution_UpdateOutputDevice<FilterRowSize,   \
                                                     FilterColSize,   \
                                                     FilterRowStride, \
                                                     FilterColStride, \
                                                     NumInputPlanes,  \
                                                     NumInputCols>
  cudaFuncSetCacheConfig(KERNEL, cudaFuncCachePreferL1);

  const int numBatches  = input.getSize(0);
  const int numInputPlanes = input.getSize(1);
  const int numInputRows = input.getSize(2);
  const int numFilters  = filters.getSize(3);

  const int optTileCol = NumInputCols;
  const int optTileBatch = 3;
  // atomicAdd on this dimension, better make sure it is well sized (k * 32)
  const int optTileFilter = std::min(256, ICC_MAX_THREADS_PER_BLOCK);
  const int optTileRow = ICC_MAX_THREADS_PER_BLOCK / optTileFilter;
  int tileCol = std::min(optTileCol, NumInputCols);
  int tileRow = std::min(optTileRow, numInputRows);
  int tileBatch = std::min(optTileBatch, numBatches);
  int tileFilter = std::min(optTileFilter, numFilters);

  // Needed for fully unrolling the batch dimension inside the kernel
  CHECK_EQ(0, tileBatch % optTileBatch);
  // Needed for fully unrolling the inputplanes dimension inside the kernel
  CHECK_EQ(0, numInputPlanes % NumInputPlanes);

  // Performs a bunch of kernel launches depending on the size of the tile
  // parameters
  for (int planeMin = 0;
       planeMin < numInputPlanes; planeMin += NumInputPlanes) {
    for (int inputCol = 0; inputCol < NumInputCols; inputCol += tileCol) {
      for (int filter = 0; filter < numFilters; filter += tileFilter) {
        //  input(batch) x input(NumInputPlanes)
        dim3 grid(cuda::ceil(numBatches, tileBatch),
                  cuda::ceil(numInputRows, tileRow),
                  1);
        dim3 block(tileFilter,
                   tileRow);

        KERNEL<<<grid, block>>>(input,
                                filters,
                                output,
                                tileRow,
                                inputCol,
                                std::min(inputCol + tileCol, NumInputCols),
                                filter,
                                std::min(filter + tileFilter, numFilters),
                                planeMin,
                                tileBatch
                               );
      }
    }
  }

  return true;
}


template <int FilterRowSize,
          int FilterColSize,
          int FilterRowStride,
          int FilterColStride,
          int NumInputPlanes,
          int NumInputCols>
bool InputCentricRelayoutConvolution_UpdateOutputRoot(
    const DeviceTensor<float, 4>& input,
    const DeviceTensor<float, 4>& filters,
    DeviceTensor<float, 4>& output) {
  // Check assumptions on data layout are enforced
  CHECK_EQ(input.getSize(0), output.getSize(0));    // numBatches
  CHECK_EQ(FilterRowSize, filters.getSize(1));          // FilterRowSize
  CHECK_EQ(FilterColSize, filters.getSize(2));          // FilterColSize
  CHECK_EQ(filters.getSize(3), output.getSize(3));  // numFilters
  CHECK_EQ(NumInputPlanes, input.getSize(1));           // NumInputPlanes
  // CHECK_EQ(NumInputRows, input.getSize(2));          // NumInputRows
  CHECK_EQ(NumInputCols, input.getSize(3));             // NumInputCols
  CHECK_EQ(input.getSize(0), output.getSize(0));     // numBatches
  // NumOutputRows
  const int ceilFilterSizeFilterStride =
    cuda::ceil(FilterRowSize, FilterRowStride);
  CHECK_EQ(output.getSize(1) - 2 * ceilFilterSizeFilterStride,
           (input.getSize(2) - FilterRowSize) / FilterRowStride + 1);
  // NumOutputCols
  CHECK_EQ(output.getSize(2),
           (input.getSize(3) - FilterColSize) / FilterColStride + 1);
  CHECK_EQ(filters.getSize(3), output.getSize(3));   // numFilters

  return
    InputCentricRelayoutConvolution_UpdateOutputHost<FilterRowSize,
                                                     FilterColSize,
                                                     FilterRowStride,
                                                     FilterColStride,
                                                     NumInputPlanes,
                                                     NumInputCols>
    (input, filters, output);
}


#define INSTANTIATE_CONVOLUTION(KernelRowSize, KernelColSize,           \
                                KernelRowStride, KernelColStride,       \
                                NumInputPlanes, NumInputCols)           \
  if (KernelRowSize == filters.getSize(1) &&                            \
      KernelColSize == filters.getSize(2) &&                            \
      KernelRowStride == filterRowStride &&                             \
      KernelColStride == filterColStride &&                             \
      NumInputPlanes == input.getSize(1) &&                             \
      NumInputCols == input.getSize(3)) {                               \
    return InputCentricRelayoutConvolution_UpdateOutputRoot<KernelRowSize, \
                                                            KernelColSize, \
                                                            KernelRowStride, \
                                                            KernelColStride, \
                                                            NumInputPlanes, \
                                                            NumInputCols> \
      (input, filters, output);                                         \
  }

bool InputCentricRelayoutConvolution_UpdateOutput(THCState* state,
                                                  THCudaTensor* inputTH,
                                                  THCudaTensor* kernelsTH,
                                                  long filterRowStride,
                                                  long filterColStride,
                                                  THCudaTensor* outputTH) {
  DeviceTensor<float, 4> input =
    torchToDeviceTensor<float, 4>(state, inputTH);
  DeviceTensor<float, 4> filters =
    torchToDeviceTensor<float, 4>(state, kernelsTH);
  DeviceTensor<float, 4> output =
    torchToDeviceTensor<float, 4>(state, outputTH);

  // 32 x 3 x 224 x 224 * 96 x 3 x 11 x 11 (s:3,3) -> 32 x 96 x 71 x 71
  // Best configuration (12.86 ms and 57% peak arithmetic) is:
  //   ICC_MAX_THREADS_PER_BLOCK = 96
  //   ICC_MIN_BLOCKS_PER_MULTIPROCESSOR = 4
  //   optTileCol = NumInputCols;
  //   optTileBatch = 3;
  //   optTileFilter = 96;
  //   optTileRow = 1;
  INSTANTIATE_CONVOLUTION(11, 11, 3, 3, 3, 224);
  INSTANTIATE_CONVOLUTION(8, 8, 2, 2, 3, 64);
  INSTANTIATE_CONVOLUTION(8, 4, 1, 2, 3, 48);

  LOG(INFO) << "InputCentricRelayoutConvolution_UpdateOutput version " <<
    " is not instantiated yet";

  return false;
}
} } } } // namespace

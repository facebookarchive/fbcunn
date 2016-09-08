// Copyright 2004-present Facebook. All Rights Reserved.

#include "src/DeviceTensorUtils.h"
#include "THCTensor.h"

#include "cuda/CudaUtils.cuh"
#include "cuda/DeviceTensor.cuh"
#include "cuda/MemoryAccess.cuh"
#include "cuda/util/CachedDeviceProperties.h"

#define ENABLE_CUDA_DEBUG
#include "cuda/CudaDebugUtils.cuh"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <glog/logging.h>

using namespace facebook::cuda;

namespace facebook { namespace deeplearning { namespace torch {

#define LOG_TARGET VLOG(1) // LOG(INFO)

template<typename T, int NumThreads, bool affine, typename ComputeT = float>
__global__ void SpatialBatchNormalizationUpdateOutputInferenceUnrolled_kernel(
    const DeviceTensor<T, 4> input,
    DeviceTensor<T, 4> output,
    DeviceTensor<T, 1> runningMean,
    DeviceTensor<T, 1> runningStddev,
    const DeviceTensor<T, 1> weight,
    const DeviceTensor<T, 1> bias) {

  static_assert(std::is_same<ComputeT, double>::value , "type");

  auto x = threadIdx.x;
  auto y = threadIdx.y;
  auto plane = blockIdx.x;
  auto batch = blockIdx.y;

  // stddev is actually 1 / stddev
  auto stddev = runningStddev[plane].ldg();
  auto mean = runningMean[plane].ldg();
  auto inp = input[batch][plane][y][x].ldg();
  if (affine) {
    // multiply with gamma and add beta
    // TODO: everyone pulling this, optimize by reusing better
    auto beta =  bias[plane].ldg();
    auto gamma = weight[plane].ldg();
    output[batch][plane][y][x] = gamma * (inp - mean) * (stddev) + beta;
  } else {
    output[batch][plane][y][x] = (inp - mean) * (stddev);
  }
}

template<typename T, int NumThreads, bool affine, typename ComputeT = float>
__global__ void SpatialBatchNormalizationUpdateOutputInference_kernel(
    const DeviceTensor<T, 4> input,
    DeviceTensor<T, 4> output,
    DeviceTensor<T, 1> runningMean,
    DeviceTensor<T, 1> runningStddev,
    const DeviceTensor<T, 1> weight,
    const DeviceTensor<T, 1> bias) {

  static_assert(std::is_same<ComputeT, double>::value , "type");

  auto x = threadIdx.x;
  auto plane = blockIdx.x;
  auto batch = blockIdx.y;

  // stddev is actually 1 / stddev
  auto stddev = runningStddev[plane].ldg();
  auto mean = runningMean[plane].ldg();
  T beta, gamma;
  if (affine) {
    beta =  bias[plane].ldg();
    gamma = weight[plane].ldg();
  }

  for (auto y = threadIdx.y; y < output.getSize(2); y += blockDim.y) {
    auto inp = input[batch][plane][y][x].ldg();
    if (affine) {
      // multiply with gamma and add beta
      // TODO: everyone pulling this, optimize by reusing better
      output[batch][plane][y][x] = gamma * (inp - mean) * (stddev) + beta;
    } else {
      output[batch][plane][y][x] = (inp - mean) * (stddev);
    }
  }

}

template<typename T, int NumThreads, bool affine, typename ComputeT = float>
__global__ void SpatialBatchNormalizationUpdateOutput_kernel(
    const DeviceTensor<T, 4> input,
    DeviceTensor<T, 4> output,
    DeviceTensor<T, 4> centered,
    DeviceTensor<T, 1> std,
    DeviceTensor<T, 4> normalized,
    DeviceTensor<T, 1> runningMean,
    DeviceTensor<T, 1> runningStddev,
    const DeviceTensor<T, 1> weight,
    const DeviceTensor<T, 1> bias,
    T epsilon,
    T momentum) {

  static_assert(std::is_same<ComputeT, double>::value , "type");

  // Assert powers of 2 for proper intra-warp shuffle reduction
  assert(blockDim.x == NumThreads);
  assert(blockDim.y == NumThreads);
  static_assert((NumThreads & (NumThreads - 1)) == 0,
                "NumThreads must be a power of 2 for proper warp shuffling");
  auto plane = blockIdx.x;
  auto numBatches = input.getSize(0);

  auto norm = (T)0;
  if (threadIdx.y == 0) {
    norm = input.getSize(0) * input.getSize(2) * input.getSize(3);
    norm = (T)1 / norm;
  }

  // 1. Compute the mean across (batch, y, x), save it and update the
  // runningMean with momentum
  auto batchMeanGlobal = (T)0;
  for (int y = threadIdx.y; y < input.getSize(2); y += NumThreads) {
    auto batchMeanLocal = (T)0;
    for (auto batch = 0; batch < numBatches; ++batch) {
      for (int x = threadIdx.x; x < input.getSize(3); x += NumThreads) {
        auto inp = (inBounds(y, x, input)) ?
          input[batch][plane][y][x].ldg() : 0.0f;
        batchMeanLocal += inp;
      }
    }
    // Reduce within warp
    for (auto i = 0; i < getMSB(NumThreads); ++i) {
      batchMeanLocal += __shfl_xor(batchMeanLocal, 1 << i, NumThreads);
    }
    // thread 0 has it
    batchMeanGlobal += batchMeanLocal;
  }

  __shared__ T shared[NumThreads];
  // thx == 0 stores into smem
  if (threadIdx.x == 0) {
    shared[threadIdx.y] = batchMeanGlobal;
  }

  __syncthreads();
  // 'transpose', and reduce within warp again
  if (threadIdx.y == 0) {
    auto batchMeanLocal = shared[threadIdx.x];
    // Reduce within warp again
    for (auto i = 0; i < getMSB(NumThreads); ++i) {
      batchMeanLocal += __shfl_xor(batchMeanLocal, 1 << i, NumThreads);
    }
    // We did an allreduce with xors, this should reduce contention on
    // shared memory.
    batchMeanGlobal = batchMeanLocal * norm;
    // Save the non momentum-altered version to share with everyone
    shared[threadIdx.x] = batchMeanGlobal;
  }
  __syncthreads();

  // Everyone picks it up
  batchMeanGlobal = shared[threadIdx.x];
  if (threadIdx.y == 0 && threadIdx.x == 0) {
    // Momentum based writeback
    runningMean[plane] =
      (1 - momentum) * runningMean[plane] + momentum * batchMeanGlobal;
  }


  // 2. Compute the stddev across (batch, y, x),
  //      save it
  //      update the runningStddev with momentum
  //      save a copy
  // All threads have the batchMean now, compute the stddev
  auto batchStddevGlobal = (T)0;
  for (int y = threadIdx.y; y < input.getSize(2); y += NumThreads) {
    auto batchStddevLocal = (T)0;
    for (auto batch = 0; batch < numBatches; ++batch) {
      for (int x = threadIdx.x; x < input.getSize(3); x += NumThreads) {
        auto inp = 0.0f;
        if (inBounds(y, x, input)) {
          inp = input[batch][plane][y][x].ldg();
          batchStddevLocal +=
            (inp - batchMeanGlobal) * (inp - batchMeanGlobal);
          centered[batch][plane][y][x] = inp - batchMeanGlobal;
        }
      }
    }
    // Reduce within warp
    for (auto i = 0; i < getMSB(NumThreads); ++i) {
      batchStddevLocal += __shfl_xor(batchStddevLocal, 1 << i, NumThreads);
    }
    // thread 0 has it
    batchStddevGlobal += batchStddevLocal;
  }

  // thx == 0 stores into smem, reuse the same smem region, be sure to kill
  // WAR / WAW dependences even if they are extremely unlikely.
  __syncthreads();
  if (threadIdx.x == 0) {
    shared[threadIdx.y] = batchStddevGlobal;
  }

  __syncthreads();
  // 'transpose', and reduce within warp again
  if (threadIdx.y == 0) {
    auto batchStddevLocal = shared[threadIdx.x];
    // Reduce within warp again
    for (auto i = 0; i < getMSB(NumThreads); ++i) {
      batchStddevLocal += __shfl_xor(batchStddevLocal, 1 << i, NumThreads);
    }
    // We did an allreduce with xors, this should reduce contention on
    // shared memory.
    batchStddevLocal *= norm;
    batchStddevGlobal = 1 / sqrt(batchStddevLocal + epsilon);
    // Save the non momentum-altered version to share with everyone
    shared[threadIdx.x] = batchStddevGlobal;
  }
  __syncthreads();

  // Everyone picks it up
  batchStddevGlobal = shared[threadIdx.x];
  // Momentum based writeback
  if (threadIdx.y == 0 && threadIdx.x == 0) {
    std[plane] = batchStddevGlobal;
    runningStddev[plane] =
      (1 - momentum) * runningStddev[plane] + momentum * batchStddevGlobal;
  }

  // Write normalized and update the output
  auto beta =  bias[plane];
  auto gamma =  weight[plane];
  for (int y = threadIdx.y; y < input.getSize(2); y += NumThreads) {
    for (int x = threadIdx.x; x < input.getSize(3); x += NumThreads) {
      if(inBounds(y, x, output)) {
        for (auto batch = 0; batch < numBatches; ++batch) {
          auto inp = input[batch][plane][y][x].ldg();
          normalized[batch][plane][y][x] =
            (inp - batchMeanGlobal) * (batchStddevGlobal);
          if (affine) {
            // multiply with gamma and add beta
            output[batch][plane][y][x] =
              gamma * (inp - batchMeanGlobal) * (batchStddevGlobal) + beta;
          } else {
            output[batch][plane][y][x] =
            (inp - batchMeanGlobal) * (batchStddevGlobal);
          }
        }
      }
    }
  }

}


template<typename T, int BatchDims, int ImageDims, bool train, bool affine, typename ComputeT = float>
void SpatialBatchNormalizationUpdateOutput(
    const DeviceTensor<T, BatchDims + ImageDims> input,
    DeviceTensor<T, BatchDims + ImageDims> output,
    DeviceTensor<T, BatchDims + ImageDims> centered,
    DeviceTensor<T, 1> std,
    DeviceTensor<T, BatchDims + ImageDims> normalized,
    DeviceTensor<T, 1> runningMean,
    DeviceTensor<T, 1> runningStddev,
    const DeviceTensor<T, 1> weight,
    const DeviceTensor<T, 1> bias,
    T epsilon,
    T momentum,
    cudaStream_t s)
{
  static_assert(BatchDims == 2, "BatchDims == 2 only atm");

  auto prop = getCurrentDeviceProperties();
  if (!train) {
    if (input.getSize(3) * input.getSize(2) < prop.maxThreadsPerBlock) {
      dim3 blocks(input.getSize(1), input.getSize(0));
      dim3 threads(input.getSize(3), input.getSize(2));
      LOG_TARGET << blocks.x << " " << blocks.y << " " << blocks.z << " "
                << threads.x << " " << threads.y << " " << threads.z;
      SpatialBatchNormalizationUpdateOutputInferenceUnrolled_kernel
        <T, 1, affine, ComputeT>
        <<<blocks, threads, 0, s>>>
        (input, output, runningMean, runningStddev, weight, bias);
    } else {
      CHECK_GE(prop.maxThreadsPerBlock, input.getSize(3)) <<
        "Need a rolled version across both threadIdx.x and y";
      dim3 blocks(input.getSize(1),
                  input.getSize(0));
      dim3 threads(input.getSize(3),
                   min(input.getSize(2),
                       floor(prop.maxThreadsPerBlock, input.getSize(3)))
                  );
      LOG_TARGET << blocks.x << " " << blocks.y << " " << blocks.z << " "
                << threads.x << " " << threads.y << " " << threads.z;
      SpatialBatchNormalizationUpdateOutputInference_kernel
        <T, 1, affine, ComputeT>
        <<<blocks, threads, 0, s>>>
        (input, output, runningMean, runningStddev, weight, bias);
    }
  } else {
    dim3 blocks(input.getSize(1));
    if (input.getSize(3) >= 16 && input.getSize(2) >= 16) {
      dim3 threads(16, 16);
      LOG_TARGET << blocks.x << " " << blocks.y << " " << blocks.z << " "
                << threads.x << " " << threads.y << " " << threads.z;
      SpatialBatchNormalizationUpdateOutput_kernel
        <T, 16, affine, ComputeT>
        <<<blocks, threads, 0, s>>>(input,
                                    output,
                                    centered,
                                    std,
                                    normalized,
                                    runningMean,
                                    runningStddev,
                                    weight,
                                    bias,
                                    epsilon,
                                    momentum);
    } else {
      dim3 threads(8, 8);
      LOG_TARGET << blocks.x << " " << blocks.y << " " << blocks.z << " "
                << threads.x << " " << threads.y << " " << threads.z;
      SpatialBatchNormalizationUpdateOutput_kernel
        <T, 8, affine, ComputeT>
        <<<blocks, threads, 0, s>>>(input,
                                    output,
                                    centered,
                                    std,
                                    normalized,
                                    runningMean,
                                    runningStddev,
                                    weight,
                                    bias,
                                    epsilon,
                                    momentum);
    }
  }

}

extern "C" void SpatialBatchNormalizationUpdateOutputFFI(
    THCState* state,
    THCudaTensor* input,
    THCudaTensor* output,
    THCudaTensor* centered,
    THCudaTensor* std,
    THCudaTensor* normalized,
    THCudaTensor* runningMean,
    THCudaTensor* runningStddev,
    THCudaTensor* weight,
    THCudaTensor* bias,
    float epsilon,
    float momentum,
    bool train,
    bool affine)
{
  // The SpatialBatchNormalization lua module is designed for
  // 4-D only: batch, plane, y, x
  constexpr int BatchDims = 2;
  constexpr int ImageDims = 2;
  typedef double ComputeT;
  if (!train) {
    if (!affine) {
      // Collapse
      SpatialBatchNormalizationUpdateOutput
        <float, BatchDims, ImageDims, false, false, ComputeT>
        (
          torchToDeviceTensor<float, BatchDims + ImageDims>(state, input),
          torchToDeviceTensor<float, BatchDims + ImageDims>(state, output),
          DeviceTensor<float, BatchDims + ImageDims>(),
          DeviceTensor<float, 1>(),
          DeviceTensor<float, BatchDims + ImageDims>(),
          torchToDeviceTensor<float, 1>(state, runningMean),
          torchToDeviceTensor<float, 1>(state, runningStddev),
          DeviceTensor<float, 1>(),
          DeviceTensor<float, 1>(),
          epsilon,
          momentum,
          THCState_getCurrentStream(state)
        );
    } else {
      // Collapse
      SpatialBatchNormalizationUpdateOutput
        <float, BatchDims, ImageDims, false, true, ComputeT>
        (
          torchToDeviceTensor<float, BatchDims + ImageDims>(state, input),
          torchToDeviceTensor<float, BatchDims + ImageDims>(state, output),
          DeviceTensor<float, BatchDims + ImageDims>(),
          DeviceTensor<float, 1>(),
          DeviceTensor<float, BatchDims + ImageDims>(),
          torchToDeviceTensor<float, 1>(state, runningMean),
          torchToDeviceTensor<float, 1>(state, runningStddev),
          torchToDeviceTensor<float, 1>(state, weight),
          torchToDeviceTensor<float, 1>(state, bias),
          epsilon,
          momentum,
          THCState_getCurrentStream(state)
        );
    }
  } else {
    if (!affine) {
      SpatialBatchNormalizationUpdateOutput
        <float, BatchDims, ImageDims, true, false, ComputeT>
      (
        torchToDeviceTensor<float, BatchDims + ImageDims>(state, input),
        torchToDeviceTensor<float, BatchDims + ImageDims>(state, output),
        torchToDeviceTensor<float, BatchDims + ImageDims>(state, centered),
        torchToDeviceTensor<float, 1>(state, std),
        torchToDeviceTensor<float, BatchDims + ImageDims>(state, normalized),
        torchToDeviceTensor<float, 1>(state, runningMean),
        torchToDeviceTensor<float, 1>(state, runningStddev),
        DeviceTensor<float, 1>(),
        DeviceTensor<float, 1>(),
        epsilon,
        momentum,
        THCState_getCurrentStream(state)
      );
    } else {
      SpatialBatchNormalizationUpdateOutput
        <float, BatchDims, ImageDims, true, true, ComputeT>
      (
        torchToDeviceTensor<float, BatchDims + ImageDims>(state, input),
        torchToDeviceTensor<float, BatchDims + ImageDims>(state, output),
        torchToDeviceTensor<float, BatchDims + ImageDims>(state, centered),
        torchToDeviceTensor<float, 1>(state, std),
        torchToDeviceTensor<float, BatchDims + ImageDims>(state, normalized),
        torchToDeviceTensor<float, 1>(state, runningMean),
        torchToDeviceTensor<float, 1>(state, runningStddev),
        torchToDeviceTensor<float, 1>(state, weight),
        torchToDeviceTensor<float, 1>(state, bias),
        epsilon,
        momentum,
        THCState_getCurrentStream(state)
      );
    }
  }

  THCudaCheck(cudaGetLastError());
}


template<typename T, int NumThreads, bool affine, typename ComputeT = float>
__global__ void SpatialBatchNormalizationUpdateGradInput_kernel(
    DeviceTensor<T, 4> gradInput,
    const DeviceTensor<T, 4> gradOutput,
    DeviceTensor<T, 4> centered,
    DeviceTensor<T, 1> std,
    const DeviceTensor<T, 1> weight) {

  static_assert(std::is_same<ComputeT, double>::value , "type");

  // Assert powers of 2 for proper intra-warp shuffle reduction
  assert(blockDim.x == NumThreads);
  assert(blockDim.y == NumThreads);
  static_assert((NumThreads & (NumThreads - 1)) == 0,
                "NumThreads must be a power of 2 for proper warp shuffling");
  auto plane = blockIdx.x;
  auto numBatches = gradInput.getSize(0);

  auto norm = (T)0;
  if (threadIdx.y == 0) {
    norm = gradInput.getSize(0) * gradInput.getSize(2) * gradInput.getSize(3);
    norm = (T)1 / norm;
  }

  // 1. Compute means across (batch, y, x)
  auto gradMeanGlobal = (T)0;
  auto centeredGradMeanGlobal = (T)0;
  for (int y = threadIdx.y; y < gradInput.getSize(2); y += NumThreads) {
    auto gradMeanLocal = (T)0;
    auto centeredGradMeanLocal = (T)0;
    for (auto batch = 0; batch < numBatches; ++batch) {
      for (int x = threadIdx.x; x < gradInput.getSize(3); x += NumThreads) {
        auto g = (inBounds(y, x, gradOutput)) ?
          gradOutput[batch][plane][y][x].ldg() : 0.0f;
        auto c = (inBounds(y, x, centered)) ?
          centered[batch][plane][y][x].ldg()   : 0.0f;
        gradMeanLocal += g;
        centeredGradMeanLocal += c * g;
      }
    }
    // Reduce within warp
    for (auto i = 0; i < getMSB(NumThreads); ++i) {
      gradMeanLocal +=
        __shfl_xor(gradMeanLocal, 1 << i, NumThreads);
      centeredGradMeanLocal +=
        __shfl_xor(centeredGradMeanLocal, 1 << i, NumThreads);
    }
    // thread 0 has it
    gradMeanGlobal += gradMeanLocal;
    centeredGradMeanGlobal += centeredGradMeanLocal;
  }

  __shared__ T shared[2][NumThreads];
  // thx == 0 stores into smem
  if (threadIdx.x == 0) {
    shared[0][threadIdx.y] = gradMeanGlobal;
    shared[1][threadIdx.y] = centeredGradMeanGlobal;
  }

  __syncthreads();
  // 'transpose', and reduce within warp again
  if (threadIdx.y == 0) {
    auto gradMeanLocal = shared[0][threadIdx.x];
    auto centeredGradMeanLocal = shared[1][threadIdx.x];
    // Reduce within warp again
    for (auto i = 0; i < getMSB(NumThreads); ++i) {
      gradMeanLocal +=
        __shfl_xor(gradMeanLocal, 1 << i, NumThreads);
      centeredGradMeanLocal +=
        __shfl_xor(centeredGradMeanLocal, 1 << i, NumThreads);
    }
    // We did an allreduce with xors, this should reduce contention on
    // shared memory.
    gradMeanGlobal = gradMeanLocal * norm;
    centeredGradMeanGlobal = centeredGradMeanLocal * norm;
    // Save the non momentum-altered version to share with everyone
    shared[0][threadIdx.x] = gradMeanGlobal;
    shared[1][threadIdx.x] = centeredGradMeanGlobal;
  }
  __syncthreads();

  // Everyone picks it up, should be broadcast into the whole gradInput
  gradMeanGlobal = shared[0][threadIdx.x];
  centeredGradMeanGlobal = shared[1][threadIdx.x];

  auto stdVal = std[plane];
  for (int y = threadIdx.y; y < gradInput.getSize(2); y += NumThreads) {
    for (auto batch = 0; batch < numBatches; ++batch) {
      for (int x = threadIdx.x; x < gradInput.getSize(3); x += NumThreads) {
        if (affine) {
          gradInput[batch][plane][y][x] =
            ( - centeredGradMeanGlobal *
                centered[batch][plane][y][x] *
                stdVal *
                stdVal
              +
                gradOutput[batch][plane][y][x]
              -
                gradMeanGlobal
            )
            * stdVal * weight[plane];
        } else {
          gradInput[batch][plane][y][x] =
            ( - centeredGradMeanGlobal *
                centered[batch][plane][y][x] *
                stdVal *
                stdVal
              +
                gradOutput[batch][plane][y][x]
              -
                gradMeanGlobal
            )
            * stdVal;
        }
      }
    }
  }

}

template<typename T, int BatchDims, int ImageDims, bool affine, typename ComputeT = float>
void SpatialBatchNormalizationUpdateGradInput(
    DeviceTensor<T, BatchDims + ImageDims> gradInput,
    const DeviceTensor<T, BatchDims + ImageDims> gradOutput,
    DeviceTensor<T, BatchDims + ImageDims> centered,
    DeviceTensor<T, 1> std,
    const DeviceTensor<T, 1> weight,
    cudaStream_t s)
{
  static_assert(BatchDims == 2, "BatchDims == 2 only atm");

  dim3 blocks(gradInput.getSize(1));
  if (gradInput.getSize(3) >= 16 && gradInput.getSize(2) >= 16) {
    dim3 threads(16, 16);
    LOG_TARGET << blocks.x << " " << blocks.y << " " << blocks.z << " "
              << threads.x << " " << threads.y << " " << threads.z;
    SpatialBatchNormalizationUpdateGradInput_kernel
      <T, 16, affine, ComputeT>
      <<<blocks, threads, 0, s>>>(gradInput,
                                  gradOutput,
                                  centered,
                                  std,
                                  weight);
  } else {
    dim3 threads(8, 8);
    LOG_TARGET << blocks.x << " " << blocks.y << " " << blocks.z << " "
              << threads.x << " " << threads.y << " " << threads.z;
    SpatialBatchNormalizationUpdateGradInput_kernel
      <T, 8, affine, ComputeT>
      <<<blocks, threads, 0, s>>>(gradInput,
                                  gradOutput,
                                  centered,
                                  std,
                                  weight);
  }

}

extern "C" void SpatialBatchNormalizationUpdateGradInputFFI(
    THCState* state,
    THCudaTensor* gradInput,
    THCudaTensor* gradOutput,
    THCudaTensor* centered,
    THCudaTensor* std,
    THCudaTensor* weight,
    bool affine) {

  // The SpatialBatchNormalization lua module is designed for
  // 4-D only: batch, plane, y, x
  constexpr int BatchDims = 2;
  constexpr int ImageDims = 2;
  typedef double ComputeT;
  if (!affine) {
    // Collapse
    SpatialBatchNormalizationUpdateGradInput
      <float, BatchDims, ImageDims, false, ComputeT>
      (
        torchToDeviceTensor<float, BatchDims + ImageDims>(state, gradInput),
        torchToDeviceTensor<float, BatchDims + ImageDims>(state, gradOutput),
        torchToDeviceTensor<float, BatchDims + ImageDims>(state, centered),
        torchToDeviceTensor<float, 1>(state, std),
        DeviceTensor<float, 1>(),
        THCState_getCurrentStream(state)
      );
  } else {
    // Collapse
    SpatialBatchNormalizationUpdateGradInput
      <float, BatchDims, ImageDims, true, ComputeT>
      (
        torchToDeviceTensor<float, BatchDims + ImageDims>(state, gradInput),
        torchToDeviceTensor<float, BatchDims + ImageDims>(state, gradOutput),
        torchToDeviceTensor<float, BatchDims + ImageDims>(state, centered),
        torchToDeviceTensor<float, 1>(state, std),
        torchToDeviceTensor<float, 1>(state, weight),
        THCState_getCurrentStream(state)
      );
  }

  THCudaCheck(cudaGetLastError());
}


template<typename T, int NumThreads, typename ComputeT = float>
__global__  void SpatialBatchNormalizationAccGradParameters_kernel(
    const DeviceTensor<T, 4> gradOutput,
    const DeviceTensor<T, 4> normalized,
    DeviceTensor<T, 1> gradWeight,
    DeviceTensor<T, 1> gradBias,
    T scale)
{

  static_assert(std::is_same<ComputeT, double>::value , "type");

  // Assert powers of 2 for proper intra-warp shuffle reduction
  assert(blockDim.x == NumThreads);
  assert(blockDim.y == NumThreads);
  static_assert((NumThreads & (NumThreads - 1)) == 0,
                "NumThreads must be a power of 2 for proper warp shuffling");
  auto plane = blockIdx.x;
  auto numBatches = gradOutput.getSize(0);

  // 1. Compute sums across (batch, y, x)
  auto gradMeanGlobal = (T)0;
  auto normalizedGradMeanGlobal = (T)0;
  for (int y = threadIdx.y; y < gradOutput.getSize(2); y += NumThreads) {
    auto gradMeanLocal = (T)0;
    auto normalizedGradMeanLocal = (T)0;
    for (auto batch = 0; batch < numBatches; ++batch) {
      for (int x = threadIdx.x; x < gradOutput.getSize(3); x += NumThreads) {
        auto g = (inBounds(y, x, gradOutput)) ?
          gradOutput[batch][plane][y][x].ldg() : 0.0f;
        auto n = (inBounds(y, x, normalized)) ?
          normalized[batch][plane][y][x].ldg() : 0.0f;
        gradMeanLocal += g;
        normalizedGradMeanLocal += n * g;
      }
    }
    // Reduce within warp
    for (auto i = 0; i < getMSB(NumThreads); ++i) {
      gradMeanLocal +=
        __shfl_xor(gradMeanLocal, 1 << i, NumThreads);
      normalizedGradMeanLocal +=
        __shfl_xor(normalizedGradMeanLocal, 1 << i, NumThreads);
    }
    // thread 0 has it
    gradMeanGlobal += gradMeanLocal;
    normalizedGradMeanGlobal += normalizedGradMeanLocal;
  }

  __shared__ T shared[2][NumThreads];
  // thx == 0 stores into smem
  if (threadIdx.x == 0) {
    shared[0][threadIdx.y] = gradMeanGlobal;
    shared[1][threadIdx.y] = normalizedGradMeanGlobal;
  }

  __syncthreads();
  // 'transpose', and reduce within warp again
  if (threadIdx.y == 0) {
    auto gradMeanLocal = shared[0][threadIdx.x];
    auto normalizedGradMeanLocal = shared[1][threadIdx.x];
    // Reduce within warp again
    for (auto i = 0; i < getMSB(NumThreads); ++i) {
      gradMeanLocal +=
        __shfl_xor(gradMeanLocal, 1 << i, NumThreads);
      normalizedGradMeanLocal +=
        __shfl_xor(normalizedGradMeanLocal, 1 << i, NumThreads);
    }
    // We did an allreduce with xors, this should reduce contention on
    // shared memory.
    gradMeanGlobal = gradMeanLocal;
    normalizedGradMeanGlobal = normalizedGradMeanLocal;

    // thread 0 has it
    if (threadIdx.x == 0) {
      gradBias[plane] += scale * gradMeanGlobal;
      gradWeight[plane] += scale * normalizedGradMeanGlobal;
    }
  }
}

template<typename T, int BatchDims, int ImageDims, typename ComputeT = float>
void SpatialBatchNormalizationAccGradParameters(
    const DeviceTensor<T, BatchDims + ImageDims> gradOutput,
    const DeviceTensor<T, BatchDims + ImageDims> normalized,
    DeviceTensor<T, 1> gradWeight,
    DeviceTensor<T, 1> gradBias,
    T scale,
    cudaStream_t s)
{
  static_assert(BatchDims == 2, "BatchDims == 2 only atm");

  dim3 blocks(gradOutput.getSize(1));
  if (gradOutput.getSize(3) >= 16 && gradOutput.getSize(2) >= 16) {
    dim3 threads(16, 16);
    LOG_TARGET << blocks.x << " " << blocks.y << " " << blocks.z << " "
              << threads.x << " " << threads.y << " " << threads.z;
    SpatialBatchNormalizationAccGradParameters_kernel<T, 16, ComputeT>
      <<<blocks, threads, 0, s>>>(gradOutput,
                                  normalized,
                                  gradWeight,
                                  gradBias,
                                  scale);
  } else {
    dim3 threads(8, 8);
    LOG_TARGET << blocks.x << " " << blocks.y << " " << blocks.z << " "
              << threads.x << " " << threads.y << " " << threads.z;
    SpatialBatchNormalizationAccGradParameters_kernel<T, 8, ComputeT>
      <<<blocks, threads, 0, s>>>(gradOutput,
                                  normalized,
                                  gradWeight,
                                  gradBias,
                                  scale);
  }

}

extern "C" void SpatialBatchNormalizationAccGradParametersFFI(
    THCState* state,
    THCudaTensor* gradOutput,
    THCudaTensor* normalized,
    THCudaTensor* gradWeight,
    THCudaTensor* gradBias,
    float scale) {
  // The SpatialBatchNormalization lua module is designed for
  // 4-D only: batch, plane, y, x
  constexpr int BatchDims = 2;
  constexpr int ImageDims = 2;
  typedef double ComputeT;
  // Collapse
  SpatialBatchNormalizationAccGradParameters
    <float, BatchDims, ImageDims, ComputeT>
    (
      torchToDeviceTensor<float, BatchDims + ImageDims>(state, gradOutput),
      torchToDeviceTensor<float, BatchDims + ImageDims>(state, normalized),
      torchToDeviceTensor<float, 1>(state, gradWeight),
      torchToDeviceTensor<float, 1>(state, gradBias),
      scale,
      THCState_getCurrentStream(state)
    );

  THCudaCheck(cudaGetLastError());
}


}}}

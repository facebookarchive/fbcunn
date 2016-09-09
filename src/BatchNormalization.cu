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

template<typename T, bool affine, typename ComputeT = float>
__global__ void BatchNormalizationUpdateOutputInferenceUnrolled_kernel(
    const DeviceTensor<T, 2> input,
    DeviceTensor<T, 2> output,
    DeviceTensor<T, 1> runningMean,
    DeviceTensor<T, 1> runningStddev,
    const DeviceTensor<T, 1> weight,
    const DeviceTensor<T, 1> bias) {

  static_assert(std::is_same<ComputeT, double>::value , "type");

  auto batch = blockIdx.y;
  auto x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x >= input.getSize(1)) {
    return;
  }

  // stddev is actually 1 / stddev
  ComputeT stddev = runningStddev[x].ldg();
  ComputeT mean = runningMean[x].ldg();
  ComputeT inp = input[batch][x].ldg();
  if (affine) {
    // multiply with gamma and add beta
    // TODO: everyone pulling this, optimize by reusing better
    ComputeT beta =  bias[x].ldg();
    ComputeT gamma = weight[x].ldg();
    output[batch][x] =  gamma * (inp - mean) * (stddev) + beta;
  } else {
    output[batch][x] = (inp - mean) * (stddev);
  }
}

template<typename T, bool affine, typename ComputeT = float>
__global__ void BatchNormalizationUpdateOutput_kernel(
    const DeviceTensor<T, 2> input,
    DeviceTensor<T, 2> output,
    DeviceTensor<T, 2> centered,
    DeviceTensor<T, 1> std,
    DeviceTensor<T, 2> normalized,
    DeviceTensor<T, 1> runningMean,
    DeviceTensor<T, 1> runningStddev,
    const DeviceTensor<T, 1> weight,
    const DeviceTensor<T, 1> bias,
    T epsilon,
    T momentum) {

  static_assert(std::is_same<ComputeT, double>::value , "type");

  auto x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x >= output.getSize(1)) {
    return;
  }

  ComputeT norm = (ComputeT)1 / input.getSize(0);

  ComputeT batchMean = (ComputeT)0;
  for (auto batch = 0; batch < output.getSize(0); ++batch) {
    ComputeT b = input[batch][x].ldg();
    batchMean += b;
  }
  batchMean *= norm;
  runningMean[x] = (1 - momentum) * runningMean[x] + momentum * batchMean;

  ComputeT stdMean = (ComputeT)0;
  for (auto batch = 0; batch < output.getSize(0); ++batch) {
    ComputeT inp = input[batch][x].ldg() ;
    centered[batch][x] = inp - batchMean;
    stdMean += (inp - batchMean) * (inp - batchMean);
  }
  stdMean =  1 / sqrt(stdMean * norm + epsilon);

  std[x] = stdMean;
  runningStddev[x] = (1 - momentum) * runningStddev[x] + momentum * stdMean;

  for (auto batch = 0; batch < output.getSize(0); ++batch) {
    output[batch][x] = centered[batch][x] * stdMean;
    normalized[batch][x] = centered[batch][x] * stdMean;
    if (affine) {
      ComputeT beta = bias[x];
      ComputeT gamma = weight[x];
      output[batch][x] = gamma * output[batch][x] + beta;
    }
  }
}


template<typename T, int BatchDims, int ImageDims, bool train, bool affine, typename ComputeT = float>
void BatchNormalizationUpdateOutput(
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
  static_assert(ImageDims == 0, "ImageDims == 0 only atm");

  dim3 threads(128);
  // auto prop = getCurrentDeviceProperties();
  if (!train) {
    dim3 blocks(ceil(input.getSize(1), 128), input.getSize(0));
    LOG_TARGET << blocks.x << " " << blocks.y << " " << blocks.z << " "
               << threads.x << " " << threads.y << " " << threads.z;
    BatchNormalizationUpdateOutputInferenceUnrolled_kernel
      <T, affine, ComputeT>
      <<<blocks, threads, 0, s>>>
      (input, output, runningMean, runningStddev, weight, bias);
  } else {
    dim3 blocks(ceil(input.getSize(1), 128));
    LOG_TARGET << blocks.x << " " << blocks.y << " " << blocks.z << " "
               << threads.x << " " << threads.y << " " << threads.z;
    BatchNormalizationUpdateOutput_kernel<T, affine, ComputeT>
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

extern "C" void BatchNormalizationUpdateOutputFFI(
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
  // The BatchNormalization lua module is designed for
  // 2-D only: batch, plane
  constexpr int BatchDims = 2;
  constexpr int ImageDims = 0;
  typedef double ComputeT;
  if (!train) {
    if (!affine) {
      // Collapse
      BatchNormalizationUpdateOutput
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
      BatchNormalizationUpdateOutput
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
      BatchNormalizationUpdateOutput
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
      BatchNormalizationUpdateOutput
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


template<typename T, bool affine, typename ComputeT = float>
__global__ void BatchNormalizationUpdateGradInput_kernel(
    DeviceTensor<T, 2> gradInput,
    const DeviceTensor<T, 2> gradOutput,
    DeviceTensor<T, 2> centered,
    DeviceTensor<T, 1> std,
    const DeviceTensor<T, 1> weight) {

  static_assert(std::is_same<ComputeT, double>::value , "type");

  auto x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x >= gradOutput.getSize(1)) {
    return;
  }

  ComputeT norm = (ComputeT)1 / gradInput.getSize(0);
  ComputeT gradMean = (ComputeT)0;
  ComputeT centeredGradMean = (ComputeT)0;
  for (auto batch = 0; batch < gradOutput.getSize(0); ++batch) {
    ComputeT g = gradOutput[batch][x].ldg();
    ComputeT c = centered[batch][x].ldg();
    gradMean += g;
    centeredGradMean += c * g;
  }
  gradMean *= norm;
  centeredGradMean *= norm;

  ComputeT stdVal = std[x];
  ComputeT weightVal = (ComputeT)0;
  if (affine) {
    weightVal = weight[x];
  }
  for (auto batch = 0; batch < gradOutput.getSize(0); ++batch) {
    if (affine) {
      gradInput[batch][x] =
        (
          - centeredGradMean * centered[batch][x] * stdVal * stdVal
          + gradOutput[batch][x]
          - gradMean
        ) * stdVal * weightVal;
    } else {
      gradInput[batch][x] =
        (
          - centeredGradMean * centered[batch][x] * stdVal * stdVal
          + gradOutput[batch][x]
          - gradMean
        ) * stdVal;
    }
  }
}

template<typename T, int BatchDims, int ImageDims, bool affine, typename ComputeT = float>
void BatchNormalizationUpdateGradInput(
    DeviceTensor<T, BatchDims + ImageDims> gradInput,
    const DeviceTensor<T, BatchDims + ImageDims> gradOutput,
    DeviceTensor<T, BatchDims + ImageDims> centered,
    DeviceTensor<T, 1> std,
    const DeviceTensor<T, 1> weight,
    cudaStream_t s)
{
  static_assert(BatchDims == 2, "BatchDims == 2 only atm");
  static_assert(ImageDims == 0, "ImageDims == 0 only atm");

  dim3 blocks(ceil(gradOutput.getSize(1), 128));
  dim3 threads(128);
  LOG_TARGET << blocks.x << " " << blocks.y << " " << blocks.z << " "
             << threads.x << " " << threads.y << " " << threads.z;
  BatchNormalizationUpdateGradInput_kernel<T, affine, ComputeT>
    <<<blocks, threads, 0, s>>>(gradInput,
                                gradOutput,
                                centered,
                                std,
                                weight);
}

extern "C" void BatchNormalizationUpdateGradInputFFI(
    THCState* state,
    THCudaTensor* gradInput,
    THCudaTensor* gradOutput,
    THCudaTensor* centered,
    THCudaTensor* std,
    THCudaTensor* weight,
    bool affine) {

  // The BatchNormalization lua module is designed for
  // 2-D only: batch, plane
  constexpr int BatchDims = 2;
  constexpr int ImageDims = 0;
  typedef double ComputeT;
  if (!affine) {
    // Collapse
    BatchNormalizationUpdateGradInput
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
    BatchNormalizationUpdateGradInput
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


template<typename T, typename ComputeT = float>
__global__  void BatchNormalizationAccGradParameters_kernel(
    const DeviceTensor<T, 2> gradOutput,
    const DeviceTensor<T, 2> normalized,
    DeviceTensor<T, 1> gradWeight,
    DeviceTensor<T, 1> gradBias,
    T scale)
{

  static_assert(std::is_same<ComputeT, double>::value , "type");

  auto x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x >= gradOutput.getSize(1)) {
    return;
  }

  ComputeT gradMean = (ComputeT)0;
  ComputeT normalizedGradMean = (ComputeT)0;
  for (auto batch = 0; batch < gradOutput.getSize(0); ++batch) {
    ComputeT g = gradOutput[batch][x].ldg();
    ComputeT n = normalized[batch][x].ldg();
    gradMean += g;
    normalizedGradMean += n * g;
  }
  gradBias[x] += scale * gradMean;
  gradWeight[x] += scale * normalizedGradMean;
}

template<typename T, int BatchDims, int ImageDims, typename ComputeT = float>
void BatchNormalizationAccGradParameters(
    const DeviceTensor<T, BatchDims + ImageDims> gradOutput,
    const DeviceTensor<T, BatchDims + ImageDims> normalized,
    DeviceTensor<T, 1> gradWeight,
    DeviceTensor<T, 1> gradBias,
    T scale,
    cudaStream_t s)
{
  static_assert(BatchDims == 2, "BatchDims == 2 only atm");
  static_assert(ImageDims == 0, "ImageDims == 0 only atm");

  dim3 blocks(ceil(gradOutput.getSize(1), 128));
  dim3 threads(128);
  LOG_TARGET << blocks.x << " " << blocks.y << " " << blocks.z << " "
             << threads.x << " " << threads.y << " " << threads.z;
  BatchNormalizationAccGradParameters_kernel<T, ComputeT>
    <<<blocks, threads, 0, s>>>(gradOutput,
                                normalized,
                                gradWeight,
                                gradBias,
                                scale);

}

extern "C" void BatchNormalizationAccGradParametersFFI(
    THCState* state,
    THCudaTensor* gradOutput,
    THCudaTensor* normalized,
    THCudaTensor* gradWeight,
    THCudaTensor* gradBias,
    float scale) {
  // The BatchNormalization lua module is designed for
  // 2-D only: batch, plane
  constexpr int BatchDims = 2;
  constexpr int ImageDims = 0;
  typedef double ComputeT;
  // Collapse
  BatchNormalizationAccGradParameters
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

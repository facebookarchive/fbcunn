/**
 * Copyright 2014 Facebook
 * @author Tudor Bosman (tudorb@fb.com)
 */

#include "CrossMapNormalization.cuh"


namespace facebook { namespace deeplearning { namespace torch {
namespace detail {

namespace {

#define INPUT(I) (input[(I) * param.featureSize])
#define INPUT_SMEM(I)  (inputSMEM[(I) * blockDim.x])
#define INPUT_SMEM_MOD(I) (INPUT_SMEM((I) % (param.kernelSize)))

// Grid is [blocks_per_image] x [images_per_batch] x 1
// Block is [threads_per_block] x 1
//
// Each thread handles one pixel, so [blocks_per_image] * [threads_per_block]
// == [image_width] x [image_height]

// Forward pass
extern __shared__ float SharedMEM[];
__global__ void updateOutput(const float* __restrict__ input,
                                   float* __restrict__ output,
                                   float* __restrict__ squaredSum,
                             CrossMapNormalizationParam param) {
  int thread = blockIdx.x * blockDim.x + threadIdx.x;
  // Only run threads that have data
  if (thread < param.featureSize) {
    // smem is laid out as [param.numFeatures][threads_per_block]
    float* inputSMEM = SharedMEM + threadIdx.x;
    int start = blockIdx.y * param.numFeatures * param.featureSize + thread;

    input      += start;
    output     += start;
    squaredSum += start;
    // Handle edge-case by padding with zeros.
    for (int i = -param.kernelRadius - 1; i < 0; ++i) {
      INPUT_SMEM(i + param.kernelSize) = 0.0f;
    }
    float runningSum = 0.0f;
    // Copy input values under the footprint to SMSM.
    for (int i = 0; i < param.kernelRadius; ++i) {
      INPUT_SMEM(i) = INPUT(i);
      runningSum += INPUT_SMEM(i) * INPUT_SMEM(i);
    }

    // Compute
#pragma unroll 4
    for (int i = 0; i < param.numFeatures - param.kernelRadius; i++) {
      runningSum -= INPUT_SMEM_MOD(i + param.kernelRadius)
                  * INPUT_SMEM_MOD(i + param.kernelRadius);
      INPUT_SMEM_MOD(i + param.kernelRadius) = INPUT(i + param.kernelRadius);
      runningSum += INPUT_SMEM_MOD(i + param.kernelRadius)
                  * INPUT_SMEM_MOD(i + param.kernelRadius);

      float result = 1.0f + param.scale * runningSum;

      *squaredSum = result;
      *output = INPUT_SMEM_MOD(i) * powf(result, -param.power);

      output     += param.featureSize;
      squaredSum += param.featureSize;
    }
    // Handle edge-case by padding with zeros.
    for (int i = param.numFeatures - param.kernelRadius;
         i < param.numFeatures; i++) {
      runningSum -= INPUT_SMEM_MOD(i + param.kernelRadius)
                  * INPUT_SMEM_MOD(i + param.kernelRadius);
      INPUT_SMEM_MOD(i + param.kernelRadius) = 0.0f;
      runningSum += INPUT_SMEM_MOD(i + param.kernelRadius)
                  * INPUT_SMEM_MOD(i + param.kernelRadius);

      float result = 1.0f + param.scale * runningSum;

      *squaredSum = result;
      *output = INPUT_SMEM_MOD(i) * powf(result, -param.power);

      output     += param.featureSize;
      squaredSum += param.featureSize;
    }
  } // if (thread < param.featureSize)
}

#define GRADOUTPUT(I) (gradOutput[(I) * param.featureSize])
#define TEMP_SMEM(I)  (gradOutputSMEM[(I) * blockDim.x])
#define TEMP_SMEM_MOD(I) (TEMP_SMEM((I) % (param.kernelSize)))

#define SQUAREDSUM(I) (squaredSum[(I) * param.featureSize])
#define SQUAREDSUM_SMEM(I)  (squaredSumSMEM[(I) * blockDim.x])
#define SQUAREDSUM_SMEM_MOD(I) (SQUAREDSUM_SMEM((I) % (param.kernelSize)))

// Backprop
__global__ void updateGradInput(const float* __restrict__ input,
                                const float* __restrict__ gradOutput,
                                const float* __restrict__ squaredSum,
                                float* __restrict__ gradInput,
                                CrossMapNormalizationParam param) {
  int thread = blockIdx.x * blockDim.x + threadIdx.x;
  // Only run threads that have data
  if (thread < param.featureSize) {
    float* inputSMEM = SharedMEM + threadIdx.x;
    const int smemBlock = blockDim.x * param.kernelSize;
    float* gradOutputSMEM = SharedMEM +     smemBlock + threadIdx.x;
    float* squaredSumSMEM = SharedMEM + 2 * smemBlock + threadIdx.x;
    int start = blockIdx.y * param.numFeatures * param.featureSize + thread;

    input      += start;
    gradOutput += start;
    squaredSum += start;
    gradInput  += start;

    // Handle edge-case by padding with zeros.
    for (int i = -param.kernelRadius - 1; i < 0; ++i) {
      INPUT_SMEM(i + param.kernelSize) = 0.0f;
      SQUAREDSUM_SMEM(i + param.kernelSize) = 1.0f;
      TEMP_SMEM(i + param.kernelSize) = 0.0f;
    }
    // Copy squaredSum values under the footprint to SMEM.
    float runningSum = 0.0f;
    for (int i = 0; i < param.kernelRadius; ++i) {
      INPUT_SMEM(i) = INPUT(i);
      SQUAREDSUM_SMEM(i) = SQUAREDSUM(i);
      TEMP_SMEM(i) = GRADOUTPUT(i) * powf(SQUAREDSUM_SMEM(i), -param.power);
      runningSum += INPUT_SMEM(i)
                  * TEMP_SMEM(i)
                  / SQUAREDSUM_SMEM(i);
    }

    // Compute
#pragma unroll 4
    for (int i = 0; i < param.numFeatures - param.kernelRadius; ++i) {
      int o = i + param.kernelRadius;
      runningSum -= INPUT_SMEM_MOD(o) * TEMP_SMEM_MOD(o)
                  / SQUAREDSUM_SMEM_MOD(o);
      INPUT_SMEM_MOD(o)      = INPUT(o);
      SQUAREDSUM_SMEM_MOD(o) = SQUAREDSUM(o);
      TEMP_SMEM_MOD(o) = GRADOUTPUT(o)
                       * powf(SQUAREDSUM_SMEM_MOD(o), -param.power);
      runningSum += INPUT_SMEM_MOD(o) * TEMP_SMEM_MOD(o)
                  / SQUAREDSUM_SMEM_MOD(o);
      *gradInput = TEMP_SMEM_MOD(i)
                 - 2 * param.power * param.scale * INPUT_SMEM_MOD(i)
                 * runningSum;
      gradInput += param.featureSize;
    }
    // Handle edge-case by padding with zeros.
    for (int i = param.numFeatures - param.kernelRadius; i < param.numFeatures;
         ++i) {
      int o = i + param.kernelRadius;
      runningSum -= INPUT_SMEM_MOD(o) * TEMP_SMEM_MOD(o)
                  / SQUAREDSUM_SMEM_MOD(o);
      INPUT_SMEM_MOD(o)      = 0.0f;
      SQUAREDSUM_SMEM_MOD(o) = 1.0f;
      TEMP_SMEM_MOD(o)       = 0.0f;
      runningSum += INPUT_SMEM_MOD(o) * TEMP_SMEM_MOD(o)
                  / SQUAREDSUM_SMEM_MOD(o);
      *gradInput = TEMP_SMEM_MOD(i)
                 - 2 * param.power * param.scale * INPUT_SMEM_MOD(i)
                 * runningSum;
      gradInput += param.featureSize;
    }
  }
}

}  // anonymous namespace

void launchCrossMapNormalizationUpdateOutputKernel(
  cudaStream_t stream,
  const float* input,
  float* output,
  float* squaredSum,
  CrossMapNormalizationParam param) {
  dim3 grid;
  dim3 block;
  block.x = 256;
  // divup
  grid.x = (param.featureSize + block.x - 1) / block.x;
  grid.y = param.batchSize;

  int smemPerBlock = block.x * param.kernelSize * sizeof(float);

  // Kernels expect adjusted params
  param.scale /= param.kernelSize;

  cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
  updateOutput<<<grid, block, smemPerBlock, stream>>>(
    input, output, squaredSum, param);
}

void launchCrossMapNormalizationUpdateGradInputKernel(
  cudaStream_t stream,
  const float* input,
  const float* gradOutput,
  const float* squaredSum,
  float* gradInput,
  CrossMapNormalizationParam param) {

  dim3 grid;
  dim3 block;
  block.x = 256;
  // divup
  grid.x = (param.featureSize + block.x - 1) / block.x;
  grid.y = param.batchSize;

  int smemPerBlock = 3 * block.x * param.kernelSize * sizeof(float);

  // Kernels expect adjusted param
  param.scale /= param.kernelSize;

  cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
  updateGradInput<<<grid, block, smemPerBlock, stream>>>(
    input, gradOutput, squaredSum, gradInput, param);
}

}}}}  // namespaces

/**
 * Copyright 2014 Facebook
 * @author Frank Jargstorff (fjargsto@fb.com)
 */

#include "LocallyConnected.cuh"
#include "cuda/CudaUtils.cuh"
#include <cassert>
#include <iostream>

using namespace facebook::cuda;

namespace facebook { namespace deeplearning { namespace torch {
namespace detail {

// Read weight tensor. Need to regularize this type of access for the
// template implementation.
//
__device__
void packOutputPlanes(float& result, DeviceTensor<float, 6>& weight,
                   int outputRow, int outputCol,
                   int kernelRow, int kernelCol,
                   int outputPlane, int inputPlane) {
  result = weight[outputRow][outputCol][kernelRow]
    [kernelCol][ outputPlane][inputPlane].ldg();
}

// Read as if weight tensor was float2 in output-plane dimension.
__device__
void packOutputPlanes(float2& result, DeviceTensor<float, 6>& weight,
                   int outputRow, int outputCol,
                   int kernelRow, int kernelCol,
                   int outputPlane, int inputPlane) {
  result.x = weight[outputRow][outputCol][kernelRow]
    [kernelCol][2 * outputPlane][inputPlane].ldg();
  result.y = weight[outputRow][outputCol][kernelRow]
    [kernelCol][2 * outputPlane + 1][inputPlane].ldg();
}

// Read as if weight tensor was float4 in output-plane dimension.
__device__
void packOutputPlanes(float4& result, DeviceTensor<float, 6>& weight,
                   int outputRow, int outputCol,
                   int kernelRow, int kernelCol,
                   int outputPlane, int inputPlane) {
  result.x = weight[outputRow][outputCol][kernelRow]
    [kernelCol][4 * outputPlane][inputPlane].ldg();
  result.y = weight[outputRow][outputCol][kernelRow]
    [kernelCol][4 * outputPlane + 1][inputPlane].ldg();
  result.z = weight[outputRow][outputCol][kernelRow]
    [kernelCol][4 * outputPlane + 2][inputPlane].ldg();
  result.w = weight[outputRow][outputCol][kernelRow]
            [kernelCol][4 * outputPlane + 3][inputPlane].ldg();
}

extern __shared__ float pShared[];

// Backprop
template <int BatchSize, typename T>
__global__ void updateGradInputBatch(DeviceTensor<T, 4> gradOutput,
                                     DeviceTensor<float, 6> weight,
                                     DeviceTensor<float, 4> gradInput,
                                     int dH, int dW) {
  // note: the "input" is being computed, i.e. "input" is the output
  int inputRow    = blockIdx.z;
  int inputCol    = blockIdx.y * blockDim.y + threadIdx.y;
  int inputPlane  = threadIdx.x / gradOutput.getSize(kPlaneDim);
  int outputPlane = threadIdx.x % gradOutput.getSize(kPlaneDim);

  int smemSize[3] = {blockDim.y, BatchSize, gradOutput.getSize(kPlaneDim)};
  DeviceTensor<T, 3> gradOutputSMEM(reinterpret_cast<T*>(pShared), smemSize);
  float vSum[BatchSize];
  if (inputCol < gradInput.getSize(kWidthDim)) { // guard right-edge
    for (int image = 0; image < BatchSize; ++image) {
      vSum[image] = 0.0f;
    }

    for (int outputRow  = max(0, (inputRow - weight.getSize(kKernelHeightDim) +
                                  dH) / dH);
         outputRow < min(inputRow / dH + 1, gradOutput.getSize(kHeightDim));
         ++outputRow) {
      for (int outputCol  = max(0, (inputCol - weight.getSize(kKernelWidthDim)
                                    + dW) / dW);
           outputCol < min(inputCol / dW + 1, gradOutput.getSize(kWidthDim));
           ++outputCol) {
        int kernelRow = inputRow - dH * outputRow;
        int kernelCol = inputCol - dW * outputCol;
        T tempWeight;
        packOutputPlanes(tempWeight, weight, outputRow, outputCol,
                         kernelRow, kernelCol, outputPlane, inputPlane);

        // use input-plane tiling to iterate images
        for (int image = inputPlane; image < BatchSize;
             image += gradInput.getSize(kPlaneDim)) {
          gradOutputSMEM[threadIdx.y][image][outputPlane] = gradOutput[image]
            [outputRow][outputCol][outputPlane];
        }
        __syncthreads();
        for (int image = 0; image < BatchSize; ++image) {
          T gradOutput = gradOutputSMEM[threadIdx.y][image][outputPlane];
          vSum[image] += dot(gradOutput, tempWeight);
        }
        __syncthreads();
      }
    }

    for (int delta = 1; delta < gradOutput.getSize(kPlaneDim); delta *= 2) {
      for (int image = 0; image < BatchSize; ++image) {
        vSum[image] += __shfl_down(vSum[image], delta);
      }
    }
    if (outputPlane == 0) {
      for (int image = 0; image < BatchSize; ++image) {
        gradInput[image][inputRow][inputCol][inputPlane] = vSum[image];
      }
    }
  } // right-edge guard
}

template <int BatchSize, typename T, int Stride>
__global__ void updateGradInputBatch(DeviceTensor<T, 4> gradOutput,
                                     DeviceTensor<float, 6> weight,
                                     DeviceTensor<float, 4> gradInput) {
  // note: the "input" is being computed, i.e. "input" is the output
  int inputRow    = blockIdx.z;
  int inputCol    = blockIdx.y * blockDim.y + threadIdx.y;
  int inputPlane  = threadIdx.x / gradOutput.getSize(kPlaneDim);
  int outputPlane = threadIdx.x % gradOutput.getSize(kPlaneDim);

  int smemSize[3] = {blockDim.y, BatchSize, gradOutput.getSize(kPlaneDim)};
  DeviceTensor<T, 3> gradOutputSMEM(reinterpret_cast<T*>(pShared), smemSize);
  float vSum[BatchSize];
  if (inputCol < gradInput.getSize(kWidthDim)) { // guard right-edge
    for (int image = 0; image < BatchSize; ++image) {
      vSum[image] = 0.0f;
    }

    for (int outputRow  = max(0, (inputRow - weight.getSize(kKernelHeightDim)
                                  + Stride) / Stride);
         outputRow < min(inputRow / Stride + 1,
                         gradOutput.getSize(kHeightDim)); ++outputRow) {
      for (int outputCol  = max(0, (inputCol - weight.getSize(kKernelWidthDim)
                                    + Stride) / Stride);
           outputCol < min(inputCol / Stride + 1,
                           gradOutput.getSize(kWidthDim)); ++outputCol) {
        int kernelRow = inputRow - Stride * outputRow;
        int kernelCol = inputCol - Stride * outputCol;
        T tempWeight;
        packOutputPlanes(tempWeight, weight, outputRow, outputCol,
                         kernelRow, kernelCol, outputPlane, inputPlane);

        // use input-plane tiling to iterate images
        for (int image = inputPlane; image < BatchSize;
             image += gradInput.getSize(kPlaneDim)) {
          gradOutputSMEM[threadIdx.y][image][outputPlane] = gradOutput[image]
            [outputRow][outputCol][outputPlane];
        }
        __syncthreads();
        for (int image = 0; image < BatchSize; ++image) {
          T gradOutput = gradOutputSMEM[threadIdx.y][image][outputPlane];
          vSum[image] += dot(gradOutput, tempWeight);
        }
        __syncthreads();
      }
    }

    for (int delta = 1; delta < gradOutput.getSize(kPlaneDim); delta *= 2) {
      for (int image = 0; image < BatchSize; ++image) {
        vSum[image] += __shfl_down(vSum[image], delta);
      }
    }
    if (outputPlane == 0) {
      for (int image = 0; image < BatchSize; ++image) {
        gradInput[image][inputRow][inputCol][inputPlane] = vSum[image];
      }
    }
  } // right-edge guard
}

template <int BatchSize, typename T, int KernelSize, int Stride>
__global__ void updateGradInputBatch(DeviceTensor<T, 4> gradOutput,
                                     DeviceTensor<float, 6> weight,
                                     DeviceTensor<float, 4> gradInput) {
  // note: the "input" is being computed, i.e. "input" is the output
  int inputRow    = blockIdx.z;
  int inputCol    = blockIdx.y * blockDim.y + threadIdx.y;
  int inputPlane  = threadIdx.x / gradOutput.getSize(kPlaneDim);
  int outputPlane = threadIdx.x % gradOutput.getSize(kPlaneDim);

  int smemSize[3] = {blockDim.y, BatchSize, gradOutput.getSize(kPlaneDim)};
  DeviceTensor<T, 3> gradOutputSMEM(reinterpret_cast<T*>(pShared), smemSize);
  float vSum[BatchSize];
  if (inputCol < gradInput.getSize(kWidthDim)) { // guard right-edge
    for (int image = 0; image < BatchSize; ++image) {
      vSum[image] = 0.0f;
    }

    for (int outputRow  = max(0, (inputRow - KernelSize + Stride) / Stride);
         outputRow < min(inputRow / Stride + 1, gradOutput.getSize(kHeightDim));
         ++outputRow) {
      for (int outputCol  = max(0, (inputCol - KernelSize + Stride) / Stride);
           outputCol < min(inputCol / Stride + 1, gradOutput.getSize(kWidthDim));
           ++outputCol) {
        int kernelRow = inputRow - Stride * outputRow;
        int kernelCol = inputCol - Stride * outputCol;
        T tempWeight;
        packOutputPlanes(tempWeight, weight, outputRow, outputCol,
                         kernelRow, kernelCol, outputPlane, inputPlane);

        // use input-plane tiling to iterate images
        for (int image = inputPlane; image < BatchSize;
             image += gradInput.getSize(kPlaneDim)) {
          gradOutputSMEM[threadIdx.y][image][outputPlane] = gradOutput[image]
            [outputRow][outputCol][outputPlane];
        }
        __syncthreads();
        for (int image = 0; image < BatchSize; ++image) {
          T gradOutput = gradOutputSMEM[threadIdx.y][image][outputPlane];
          vSum[image] += dot(gradOutput, tempWeight);
        }
        __syncthreads();
      }
    }

    for (int delta = 1; delta < gradOutput.getSize(kPlaneDim); delta *= 2) {
      for (int image = 0; image < BatchSize; ++image) {
        vSum[image] += __shfl_down(vSum[image], delta);
      }
    }
    if (outputPlane == 0) {
      for (int image = 0; image < BatchSize; ++image) {
        gradInput[image][inputRow][inputCol][inputPlane] = vSum[image];
      }
    }
  } // right-edge guard
}

// Iterating kernel.
//
template <int BatchSize, typename T>
__global__ void updateGradInputBatch(DeviceTensor<T, 4> gradOutput,
                                     DeviceTensor<float, 6> weight,
                                     DeviceTensor<float, 4> gradInput,
                                     int dH, int dW,
                                     int gradOutputPlaneThreads) {
  // note: the "input" is being computed, i.e. "input" is the output
  int inputRow     = blockIdx.z;
  int inputCol     = blockIdx.y * blockDim.y + threadIdx.y;
  int inputPlane   = threadIdx.x / gradOutputPlaneThreads;
  int outputThread = threadIdx.x % gradOutputPlaneThreads;

  float vSum[BatchSize];
  if (inputCol < gradInput.getSize(kWidthDim)) { // guard right-edge
    for (int image = 0; image < BatchSize; ++image) {
      vSum[image] = 0.0f;
    }

    for (int outputRow  = max(0, (inputRow - weight.getSize(kKernelHeightDim) +
                                  dH) / dH);
         outputRow < min(inputRow / dH + 1, gradOutput.getSize(kHeightDim));
         ++outputRow) {
      for (int outputCol  = max(0, (inputCol - weight.getSize(kKernelWidthDim)
                                    + dW) / dW);
           outputCol < min(inputCol / dW + 1, gradOutput.getSize(kWidthDim));
           ++outputCol) {
        int kernelRow = inputRow - dH * outputRow;
        int kernelCol = inputCol - dW * outputCol;
        for (int outputPlane = outputThread;
             outputPlane < gradOutput.getSize(kPlaneDim);
             outputPlane += gradOutputPlaneThreads) {
          T tempWeight;
          packOutputPlanes(tempWeight, weight, outputRow, outputCol,
                           kernelRow, kernelCol, outputPlane, inputPlane);

          for (int image = 0; image < BatchSize; ++image) {
            T gradOut = gradOutput[image][outputRow][outputCol]
              [outputPlane];

            vSum[image] += dot(gradOut, tempWeight);
          }
        }
      }
    }

    for (int delta = 1; delta < gradOutputPlaneThreads; delta *= 2) {
      for (int image = 0; image < BatchSize; ++image) {
        vSum[image] += __shfl_down(vSum[image], delta);
      }
    }
    if (outputThread == 0) {
      for (int image = 0; image < BatchSize; ++image) {
        gradInput[image][inputRow][inputCol][inputPlane] = vSum[image];
      }
    }
  } // right-edge guard
}


#define UPDATE_GRAD_INPUT_SIZE_STRIDE(SIZE, STRIDE)     case SIZE:      \
  updateGradInputBatch<BatchSize, T, SIZE, STRIDE><<<grid, block, smem, \
                                                     stream>>>(         \
    gradOutput, weight, gradInput);                                     \
break

// Dispatch based on input- and output-planes being powers of two
// in which case an optimized version of the kernel can be used.
//
template <int BatchSize, typename T>
void
dispatchUpdateGradInputPlanePOT(cudaStream_t stream,
                                DeviceTensor<T, 4> gradOutput,
                                DeviceTensor<float, 6> weight,
                                DeviceTensor<float, 4> gradInput,
                                int dH, int dW) {
  const int kBlockSize = 256;

  int gradOutPlaneThreads = kBlockSize / gradInput.getSize(kPlaneDim);
  if (gradOutPlaneThreads < gradOutput.getSize(kPlaneDim) ||
      !isPowerOfTwo(gradInput.getSize(kPlaneDim)) ||
      !isPowerOfTwo(gradOutput.getSize(kPlaneDim)) ||
      gradOutput.getSize(kPlaneDim) > 32) {
    // gradOutPlaneThreads must be a power of two or multiple of 32.
    gradOutPlaneThreads = std::min(32, gradOutPlaneThreads);
    gradOutPlaneThreads = greatestPowerOfTwoLessEq(gradOutPlaneThreads);

    dim3 block(gradInput.getSize(kPlaneDim) * gradOutPlaneThreads);
    dim3 grid(1, gradInput.getSize(kWidthDim), gradInput.getSize(kHeightDim));

    updateGradInputBatch<BatchSize, T><<<grid, block,
      0, stream>>>(gradOutput, weight,
                   gradInput, dH, dW,
                   gradOutPlaneThreads);
  } else {
    int totalPlanes = gradOutput.getSize(kPlaneDim)
      * gradInput.getSize(kPlaneDim);
    dim3 block(totalPlanes, kBlockSize / totalPlanes);
    dim3 grid(1,
              cuda::ceil(gradInput.getSize(kWidthDim),
                         static_cast<int>(block.y)),
              gradInput.getSize(kHeightDim));
    int smem = block.y * BatchSize * gradOutput.getSize(kPlaneDim)
      * sizeof(float4);
    // small-kernel optimization
    if (weight.getSize(kKernelWidthDim) == weight.getSize(kKernelHeightDim) &&
        dH == dW) {
      switch (dH) {
        case 1:
        {
          switch (weight.getSize(kKernelWidthDim)) {
            UPDATE_GRAD_INPUT_SIZE_STRIDE(3, 1);
            UPDATE_GRAD_INPUT_SIZE_STRIDE(5, 1);
            default:
              updateGradInputBatch<BatchSize, T, 1><<<grid, block,
                smem, stream>>>(
                  gradOutput, weight, gradInput);
          }
        }
        break;
        case 2:
        {
          switch (weight.getSize(kKernelWidthDim)) {
            UPDATE_GRAD_INPUT_SIZE_STRIDE(3, 2);
            UPDATE_GRAD_INPUT_SIZE_STRIDE(7, 2);
            default:
              updateGradInputBatch<BatchSize, T, 2><<<grid, block,
                smem, stream>>>(
                  gradOutput, weight, gradInput);
          }
        }
        break;
        default:
          updateGradInputBatch<BatchSize, T><<<grid, block,
            smem, stream>>>(gradOutput,
                            weight,
                            gradInput,
                            dH, dW);
      }
    } else {
      updateGradInputBatch<BatchSize, T><<<grid, block,
        smem, stream>>>(gradOutput,
                        weight,
                        gradInput,
                        dH, dW);
    }
  }
}

template <int BatchSize>
void
dispatchUpdateGradInputBatchIPT(cudaStream_t stream,
                                DeviceTensor<float, 4> gradOutput,
                                DeviceTensor<float, 6> weight,
                                DeviceTensor<float, 4> gradInput,
                                int dH, int dW) {
  if (gradOutput.getSize(kPlaneDim) % 4 == 0 &&
      isAligned(gradOutput.data(), sizeof(float4)) &&
      kFloat4Optimization) {
    // create float4 based gradOutput tensor
    DeviceTensor<float4, 4> gradOutput4 = convertImageBatch<float4>(gradOutput);

    dispatchUpdateGradInputPlanePOT<BatchSize, float4>(
      stream, gradOutput4, weight, gradInput, dH, dW);
  } else if (gradOutput.getSize(kPlaneDim) % 2 == 0 &&
             isAligned(gradOutput.data(), sizeof(float2)) &&
             kFloat2Optimization) {
    // create float2 based gradOutput tensor
    DeviceTensor<float2, 4> gradOutput2 = convertImageBatch<float2>(gradOutput);

    dispatchUpdateGradInputPlanePOT<BatchSize, float2>(
      stream, gradOutput2, weight, gradInput, dH, dW);
  } else {
    dispatchUpdateGradInputPlanePOT<BatchSize, float>(
      stream, gradOutput, weight, gradInput, dH, dW);
  }
}

#define UPDATE_GRAD_INPUT_CASE(B)     case B:                           \
  dispatchUpdateGradInputBatchIPT<B>(                                   \
    stream, gradOutput, weight, gradInput, dH, dW);                      \
  break


void updateGradInputBatchPOT(cudaStream_t stream,
                             DeviceTensor<float, 4> gradOutput,
                             DeviceTensor<float, 6> weight,
                             DeviceTensor<float, 4> gradInput,
                             int batchSize, int dH, int dW) {
  switch (batchSize) {
    UPDATE_GRAD_INPUT_CASE(128);
    UPDATE_GRAD_INPUT_CASE(64);
    UPDATE_GRAD_INPUT_CASE(32);
    UPDATE_GRAD_INPUT_CASE(16);
    UPDATE_GRAD_INPUT_CASE(8);
    UPDATE_GRAD_INPUT_CASE(4);
    UPDATE_GRAD_INPUT_CASE(2);
    UPDATE_GRAD_INPUT_CASE(1);
    default:
      assert(false); // input validation, for debugging only
  }
}

void locallyConnectedUpdateGradInput(cudaStream_t stream,
                                     const float* gradOutput,
                                     const float* weight,
                                     float* gradInput,
                                     LocallyConnectedParam& params) {
  long batchIdx = 0;

  int weightSize[6] = {params.outputHeight, params.outputWidth,
    params.kernelHeight, params.kernelWidth,
    params.outputPlanes, params.inputPlanes};
  DeviceTensor<float, 6> cudaWeight(const_cast<float*>(weight), weightSize);

  int batchSize = 16;
  int inputSize[4] = {batchSize, params.inputHeight, params.inputWidth,
    params.inputPlanes};
  int outputSize[4] = {batchSize, params.outputHeight, params.outputWidth,
    params.outputPlanes};
  while (batchSize > 0) {
    while (batchIdx < (params.batchSize / batchSize) * batchSize) {
      DeviceTensor<float, 4> cudaGradOutput(const_cast<float*>(gradOutput),
                                            outputSize);
      DeviceTensor<float, 4> cudaGradInput(gradInput, inputSize);
      updateGradInputBatchPOT(stream, cudaGradOutput, cudaWeight, cudaGradInput,
                              batchSize, params.dH, params.dW);
      batchIdx += batchSize;
      gradOutput += cudaGradOutput.numElements();
      gradInput += cudaGradInput.numElements();
    }
    batchSize /= 2;
    inputSize[0] = batchSize;
    outputSize[0] = batchSize;
  }
}


} // detail namespace

}}}  // namespaces

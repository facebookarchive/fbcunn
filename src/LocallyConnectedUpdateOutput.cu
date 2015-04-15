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

extern __shared__ float pShared[];

template <int BatchSize, typename T>
__launch_bounds__(256, 6)
__global__ void updateOutputBatch(DeviceTensor<T, 4> input,
                                  DeviceTensor<T, 6> weight,
                                  DeviceTensor<float, 3> bias,
                                  DeviceTensor<float, 4> output,
                                  int dH, int dW) {
  int outputRow   = blockIdx.z;
  int outputCol   = blockIdx.y * blockDim.y + threadIdx.y;
  int outputPlane = threadIdx.x / input.getSize(kPlaneDim);
  int inputRow    = outputRow * dH;
  int inputCol    = outputCol * dW;
  int inputPlane  = threadIdx.x % input.getSize(kPlaneDim);

  int inputSizeSMEM[3] = {blockDim.y, BatchSize, input.getSize(kPlaneDim)};
  DeviceTensor<T, 3> inputSMEM(reinterpret_cast<T*>(pShared), inputSizeSMEM);
  // compute offset to end of inputSMEM
  int offsetSMEM = inputSizeSMEM[0] * inputSizeSMEM[1] * inputSizeSMEM[2]
    * sizeof(T) / sizeof(float);
  // create shared memory bias tensor
  int biasSizeSMEM[2] = {blockDim.y, output.getSize(kPlaneDim)};
  DeviceTensor<float, 2> biasSMEM(pShared + offsetSMEM, biasSizeSMEM);

  float vSum[BatchSize];
  if (outputCol < output.getSize(kWidthDim)) { // guard right-edge
    // stage biases into shared
    if (threadIdx.x < output.getSize(kPlaneDim)) {
      biasSMEM[threadIdx.y][threadIdx.x] = bias[outputRow][outputCol]
        [threadIdx.x];
    }
    for (int batch = 0; batch < BatchSize; ++batch) {
      vSum[batch] = 0.0f;
    }

    for (int kernelRow = 0; kernelRow < weight.getSize(kKernelHeightDim);
         ++kernelRow) {
      for (int kernelCol = 0; kernelCol < weight.getSize(kKernelWidthDim);
           ++kernelCol) {
        T w = weight[outputRow][outputCol][kernelRow][kernelCol]
          [outputPlane][inputPlane];
        // use output-plane tiling to iterate images
        for (int image = outputPlane; image < BatchSize;
             image += output.getSize(kPlaneDim)) {
          inputSMEM[threadIdx.y][image][inputPlane] = input[image]
            [inputRow + kernelRow][inputCol + kernelCol][inputPlane].ldg();
        }
        __syncthreads();
        for (int image = 0; image < BatchSize; ++image) {
          T in = inputSMEM[threadIdx.y][image][inputPlane];
          vSum[image] += dot(in, w);
        }
        __syncthreads();
      }
    }
    for (int delta = 1; delta < input.getSize(kPlaneDim); delta *= 2) {
      for (int batch = 0; batch < BatchSize; ++batch) {
        vSum[batch] += __shfl_down(vSum[batch], delta);
      }
    }

    if (inputPlane == 0) {
      for (int batch = 0; batch < BatchSize; ++batch) {
        output[batch][outputRow][outputCol][outputPlane] =
          vSum[batch] + biasSMEM[threadIdx.y][outputPlane];
      }
    }
  } // right-edge guard
}

template <int BatchSize, typename T>
__global__ void updateOutputBatch(DeviceTensor<T, 4> input,
                                  DeviceTensor<T, 6> weight,
                                  DeviceTensor<float, 3> bias,
                                  DeviceTensor<float, 4> output,
                                  int dH, int dW, int inputPlaneThreads) {
  int outputRow   = blockIdx.z;
  int outputCol   = blockIdx.y * blockDim.y + threadIdx.y;
  int outputPlane = threadIdx.x / inputPlaneThreads;
  int inputRow    = outputRow * dH;
  int inputCol    = outputCol * dW;
  int inputThread = threadIdx.x % inputPlaneThreads;

  float vSum[BatchSize];
  if (outputCol < output.getSize(kWidthDim)) { // guard right-edge
    for (int batch = 0; batch < BatchSize; ++batch) {
      vSum[batch] = 0.0f;
    }

    for (int kernelRow = 0; kernelRow < weight.getSize(kKernelHeightDim);
         ++kernelRow) {
      for (int kernelCol = 0; kernelCol < weight.getSize(kKernelWidthDim);
           ++kernelCol) {
        for (int inputPlane = inputThread;
             inputPlane < input.getSize(kPlaneDim);
             inputPlane += inputPlaneThreads) {
          T w = weight[outputRow][outputCol][kernelRow][kernelCol]
            [outputPlane][inputPlane];
          for (int batch = 0; batch < BatchSize; ++batch) {
            T in = input[batch][inputRow + kernelRow]
              [inputCol + kernelCol][inputPlane];
            vSum[batch] += dot(in, w);
          }
        }
      }
    }
    for (int delta = 1; delta < inputPlaneThreads; delta *= 2) {
      for (int batch = 0; batch < BatchSize; ++batch) {
        vSum[batch] += __shfl_down(vSum[batch], delta);
      }
    }

    if (inputThread == 0) {
      for (int batch = 0; batch < BatchSize; ++batch) {
        output[batch][outputRow][outputCol][outputPlane] =
          vSum[batch] + bias[outputRow][outputCol][outputPlane];
      }
    }
  } // right-edge guard
}

// Dispatch based on input- and output-planes being powers of two
// in which case an optimized version of the kernel can be used.
//
template <int BatchSize, typename T>
void
dispatchUpdateOutputPlanePOT(cudaStream_t stream,
                             DeviceTensor<T, 4> input,
                             DeviceTensor<T, 6> weight,
                             DeviceTensor<float, 3> bias,
                             DeviceTensor<float, 4> output,
                             int dH, int dW) {
  const int kBlockSize = 256; // threads
  int inputPlaneThreads = kBlockSize / output.getSize(kPlaneDim);
  // The following conditions force the catch-all slow path:
  //
  // inputPlaneThreads < inputPlanes: This condition indicates that the
  // total number of threads per pixel column (i.e.
  // outputPlanes * inputPlanes) is greater than kBlockSize. In this case
  // the iterating kernel is necessary to cover all inputs by looping.
  //
  // outputPlanes and inputPlanes must be powers of two: inputPlanes must
  // be a power of two for the in-warp shuffle reductions to work. In order
  // for the input plane threads to properly align (when multiple columns
  // are processed by a single CTA), outputPlanes must also be a power of two.
  //
  // inputPlanes > 32: All input-plane threads must belong to the same warp
  // in order for in-warp shuffle reductions to work.
  if (inputPlaneThreads < input.getSize(kPlaneDim) ||
      !isPowerOfTwo(output.getSize(kPlaneDim)) ||
      !isPowerOfTwo(input.getSize(kPlaneDim)) ||
      input.getSize(kPlaneDim) > 32) {
    inputPlaneThreads = std::min(32, inputPlaneThreads);
    inputPlaneThreads = greatestPowerOfTwoLessEq(inputPlaneThreads);

    dim3 block(output.getSize(kPlaneDim) * inputPlaneThreads);
    dim3 grid(1, output.getSize(kWidthDim), output.getSize(kHeightDim));
    updateOutputBatch<BatchSize, T><<<grid, block, 0, stream>>>(
      input, weight, bias,
      output, dH, dW,
      inputPlaneThreads);
  } else {
    const int totalPlanes =
      input.getSize(kPlaneDim) * output.getSize(kPlaneDim);
    dim3 block(totalPlanes, kBlockSize / totalPlanes);
    dim3 grid(1,
              cuda::ceil(output.getSize(kWidthDim),
                         static_cast<int>(block.y)),
              output.getSize(kHeightDim));
    // smem for input caching
    int smem = block.y * BatchSize * input.getSize(kPlaneDim) * sizeof(T);
    // smem for bias caching
    smem += block.y * output.getSize(kPlaneDim) * sizeof(float);
    updateOutputBatch<BatchSize, T><<<grid, block, smem, stream>>>(
      input, weight,
      bias, output,
      dH, dW);
  }
}

// Dispatch updateOutput implementations depending on the possible degree
// of in-thread-parallelism.
template <int BatchSize>
void
dispatchUpdateOutputITP(cudaStream_t stream,
                        DeviceTensor<float, 4> input,
                        DeviceTensor<float, 6> weight,
                        DeviceTensor<float, 3> bias,
                        DeviceTensor<float, 4> output,
                        int dH, int dW) {
  // determine if float4 based (16-byte) data reading is possible
  if (input.getSize(kPlaneDim) % 4 == 0 &&
      isAligned(input.data(), sizeof(float4)) &&
      kFloat4Optimization) {

    // create float4 based input tensor
    DeviceTensor<float4, 4> input4 = convertImageBatch<float4>(input);
    // creat float4 based weight tensor
    DeviceTensor<float4, 6> weight4 = convertWeight<float4>(weight);

    dispatchUpdateOutputPlanePOT<BatchSize, float4>(
      stream, input4, weight4, bias, output, dH, dW);
    // determine if float2 based (8-byte) data reading is possible
  } else if (input.getSize(kPlaneDim) % 2 == 0 &&
             isAligned(input.data(), sizeof(float2)) &&
             kFloat2Optimization) {
    // create float2 based input tensor
    DeviceTensor<float2, 4> input2 = convertImageBatch<float2>(input);
    // creat float2 based weight tensor
    DeviceTensor<float2, 6> weight2 = convertWeight<float2>(weight);

    dispatchUpdateOutputPlanePOT<BatchSize, float2>(
      stream, input2, weight2, bias,
      output, dH, dW);
  } else {
    dispatchUpdateOutputPlanePOT<BatchSize, float>(
      stream, input, weight, bias,
      output, dH, dW);
  }
}

#define UPDATE_OUTPUT_CASE(BatchSize)     case BatchSize:               \
  dispatchUpdateOutputITP<BatchSize>(stream, input, weight, bias, output, \
                                     dH, dW);                           \
  break

// Dispatcher function that binds the batchSize, which must be a power-of-two
// (POT) to a function template with the batch size baked in.
void updateOutputBatchPOT(cudaStream_t stream,
                          DeviceTensor<float, 4> input,
                          DeviceTensor<float, 6> weight,
                          DeviceTensor<float, 3> bias,
                          DeviceTensor<float, 4> output,
                          int batchSize, int dH, int dW) {
  switch (batchSize) {
    UPDATE_OUTPUT_CASE(128);
    UPDATE_OUTPUT_CASE(64);
    UPDATE_OUTPUT_CASE(32);
    UPDATE_OUTPUT_CASE(16);
    UPDATE_OUTPUT_CASE(8);
    UPDATE_OUTPUT_CASE(4);
    UPDATE_OUTPUT_CASE(2);
    UPDATE_OUTPUT_CASE(1);
    default:
      assert(false); // input validation, for debugging only
  }
}

void locallyConnectedUpdateOutput(cudaStream_t stream,
                                  const float* input, const float* weight,
                                  const float* bias, float* output,
                                  LocallyConnectedParam& params) {
  int weightSize[6] = {params.outputHeight, params.outputWidth,
    params.kernelHeight, params.kernelWidth,
    params.outputPlanes, params.inputPlanes};
  DeviceTensor<float, 6> cudaWeight(const_cast<float*>(weight), weightSize);
  int biasSize[3] = {params.outputHeight, params.outputWidth,
                     params.outputPlanes};
  DeviceTensor<float, 3> cudaBias(const_cast<float*>(bias), biasSize);

  long batchIdx = 0;
  // Iterate images in the batch; processing successively smaller
  // sub-batches.
  // maxBatchSize is the biggest sub-batch size. this should be picked
  // based on the performance of the underlying kernels for each batch
  // size. Performance of the kernels increases with batch size up to a
  // point. Sub batch sizes must be powers of two.
  int maxBatchSize = 16;
  int inputSize[4] = {maxBatchSize, params.inputHeight, params.inputWidth,
                      params.inputPlanes};
  int outputSize[4] = {maxBatchSize, params.outputHeight,
                       params.outputWidth, params.outputPlanes};
  while (maxBatchSize > 0) {
    while (batchIdx < (params.batchSize / maxBatchSize) * maxBatchSize) {
      DeviceTensor<float, 4> cudaInput(const_cast<float*>(input), inputSize);
      DeviceTensor<float, 4> cudaOutput(output, outputSize);
      updateOutputBatchPOT(stream, cudaInput, cudaWeight, cudaBias,
                           cudaOutput, maxBatchSize,
                           params.dH, params.dW);
      batchIdx += maxBatchSize;
      input += cudaInput.numElements();
      output += cudaOutput.numElements();
    }
    maxBatchSize /= 2;
    inputSize[0] = maxBatchSize;
    outputSize[0] = maxBatchSize;
  }
}

} // detail namespace

}}}  // namespaces

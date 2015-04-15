/**
 * Copyright 2014 Facebook
 * @author Frank Jargstorff (fjargsto@fb.com)
 */

#include "LocallyConnected.cuh"
#include "cuda/CudaUtils.cuh"
#include <cassert>

using namespace facebook::cuda;

namespace facebook { namespace deeplearning { namespace torch {
namespace detail {

template <int BatchSize, typename T>
__launch_bounds__(256, 6)
__global__ void accGradWeight(DeviceTensor<T, 4> input,
                              DeviceTensor<float, 4> gradOutput,
                              DeviceTensor<T, 6> gradWeight,
                              float scale,
                              int dH, int dW) {
  int outputRow   = blockIdx.z;
  int outputCol   = blockIdx.y * blockDim.y + threadIdx.y;
  int outputPlane = threadIdx.x / input.getSize(kPlaneDim);
  int inputRow    = outputRow * dH;
  int inputCol    = outputCol * dW;
  int inputPlane  = threadIdx.x % input.getSize(kPlaneDim);

  int smemSize[3] = {blockDim.y, BatchSize, gradOutput.getSize(kPlaneDim)};
  extern __shared__ float pShared[];
  DeviceTensor<float, 3> gradOutputSMEM(pShared + 4, smemSize);

  if (outputCol < gradOutput.getSize(kWidthDim)) {
    {
      int outputPlaneT = threadIdx.x % gradOutput.getSize(kPlaneDim);
      int inputPlaneT  = threadIdx.x / gradOutput.getSize(kPlaneDim);
      for (int image = inputPlaneT; image < BatchSize;
           image += input.getSize(kPlaneDim)) {
        gradOutputSMEM[threadIdx.y][image][outputPlaneT] = gradOutput[image]
          [outputRow][outputCol][outputPlaneT];
      }
    }
    __syncthreads();
    for (int kernelRow = 0; kernelRow < gradWeight.getSize(kKernelHeightDim);
         ++kernelRow) {
      for (int kernelCol = 0; kernelCol < gradWeight.getSize(kKernelWidthDim);
           ++kernelCol) {
        T sum;
        zero(sum);
        for (int image = 0; image < BatchSize; ++image) {
          float gradOut = gradOutputSMEM[threadIdx.y][image][outputPlane];
          T in = input[image][inputRow + kernelRow][inputCol + kernelCol]
            [inputPlane].ldg();
          sum += gradOut * in;
        }
        T gw = gradWeight[outputRow][outputCol][kernelRow]
          [kernelCol][outputPlane][inputPlane];
        gw += scale * sum;
        gradWeight[outputRow][outputCol][kernelRow][kernelCol]
          [outputPlane][inputPlane] = gw;
      }
    }
  }
}

template <int BatchSize, typename T>
__global__ void accGradWeight(DeviceTensor<T, 4> input,
                              DeviceTensor<float, 4> gradOutput,
                              DeviceTensor<T, 6> gradWeight,
                              float scale,
                              int dH, int dW, int inputPlaneThreads) {
  int outputRow   = blockIdx.z;
  int outputCol   = blockIdx.y * blockDim.y + threadIdx.y;
  int outputPlane = threadIdx.x / inputPlaneThreads;
  int inputRow    = outputRow * dH;
  int inputCol    = outputCol * dW;
  int inputThread  = threadIdx.x % inputPlaneThreads;

  if (outputCol < gradOutput.getSize(kWidthDim)) {
    for (int kernelRow = 0; kernelRow < gradWeight.getSize(kKernelHeightDim);
         ++kernelRow) {
      for (int kernelCol = 0; kernelCol < gradWeight.getSize(kKernelWidthDim);
           ++kernelCol) {
        for (int inputPlane = inputThread;
             inputPlane < input.getSize(kPlaneDim);
             inputPlane += inputPlaneThreads) {
          T sum;
          zero(sum);
          for (int image = 0; image < BatchSize; ++image) {
            float gradOut = gradOutput[image][outputRow][outputCol]
              [outputPlane].ldg();
            T in = input[image][inputRow + kernelRow]
              [inputCol + kernelCol][inputPlane].ldg();
            sum += gradOut * in;
          }
          T gw = gradWeight[outputRow][outputCol][kernelRow]
            [kernelCol][outputPlane][inputPlane];
          gw += scale * sum;
          gradWeight[outputRow][outputCol][kernelRow][kernelCol]
            [outputPlane][inputPlane] = gw;
        }
      }
    }
  }
}

// Dispatch based on input- and output-planes being powers of two
// in which case an optimized version of the kernel can be used.
//
template <int BatchSize, typename T>
void
dispatchAccGradWeightsPlanePOT(cudaStream_t stream,
                               DeviceTensor<T, 4> input,
                               DeviceTensor<float, 4> gradOutput,
                               DeviceTensor<T, 6> gradWeight,
                               float scale,
                               int dH, int dW) {
  const int kBlockSize = 256; // threads
  int inputPlaneThreads = kBlockSize / gradOutput.getSize(kPlaneDim);
  if (inputPlaneThreads < input.getSize(kPlaneDim)) {
    // inputPlaneThreads must be a power of two and not greater 32.
    inputPlaneThreads = std::min(32, inputPlaneThreads);
    inputPlaneThreads = greatestPowerOfTwoLessEq(inputPlaneThreads);

    dim3 block(gradOutput.getSize(kPlaneDim) * inputPlaneThreads);
    dim3 grid(1, gradOutput.getSize(kWidthDim),
              gradOutput.getSize(kHeightDim));
    accGradWeight<BatchSize, T><<<grid, block, 0, stream>>>(
      input, gradOutput,
      gradWeight, scale, dH, dW,
      inputPlaneThreads);
  } else  {
    int totalPlanes = gradOutput.getSize(kPlaneDim)
      * input.getSize(kPlaneDim);
    dim3 block(totalPlanes, kBlockSize / totalPlanes);
    dim3 grid(1, cuda::ceil(gradOutput.getSize(kWidthDim),
                            static_cast<int>(block.y)),
              gradOutput.getSize(kHeightDim));
    const int smem = (block.y * BatchSize * gradOutput.getSize(kPlaneDim)
                      + 4) * sizeof(float);
    accGradWeight<BatchSize, T><<<grid, block, smem, stream>>>(
      input, gradOutput,
      gradWeight, scale,
      dH, dW);
  }
}

// Dispatch accGradWeight implementations depending on the possible degree
// of in-thread-parallelism.
template <int BatchSize>
void
dispatchAccGradWeightsITP(cudaStream_t stream,
                          DeviceTensor<float, 4> input,
                          DeviceTensor<float, 4> gradOutput,
                          DeviceTensor<float, 6> gradWeight,
                          float scale,
                          int dH, int dW) {
  // determine if float4 based data IO is possible
  if (input.getSize(kPlaneDim) % 4 == 0 &&
      isAligned(input.data(), sizeof(float4)) &&
      kFloat4Optimization) {
    // create float4 based input tensor
    DeviceTensor<float4, 4> input4 = convertImageBatch<float4>(input);
    // creat float4 based weight tensor
    DeviceTensor<float4, 6> gradWeight4 = convertWeight<float4>(gradWeight);

    dispatchAccGradWeightsPlanePOT<BatchSize, float4>(
      stream, input4, gradOutput,
      gradWeight4, scale,
      dH, dW);
  } else if (input.getSize(kPlaneDim) % 2 == 0 &&
      isAligned(input.data(), sizeof(float2)) && kFloat2Optimization) {
    // create float2 based input tensor
    DeviceTensor<float2, 4> input2 = convertImageBatch<float2>(input);
    // creat float2 based weight tensor
    DeviceTensor<float2, 6> gradWeight2 = convertWeight<float2>(gradWeight);

    dispatchAccGradWeightsPlanePOT<BatchSize, float2>(
      stream, input2, gradOutput,
      gradWeight2, scale,
      dH, dW);
  } else {
    dispatchAccGradWeightsPlanePOT<BatchSize, float>(
      stream, input, gradOutput,
      gradWeight, scale,
      dH, dW);
  }
}

// -----------------------------------------------------------------------------
// Bias
//

template <int BatchSize>
__global__ void accGradBiasBatch(DeviceTensor<float, 4> gradOutput,
                                 DeviceTensor<float, 3> gradBias,
                                 float scale) {
  int outputCol   = blockIdx.y * blockDim.y + threadIdx.y;
  int outputRow   = blockIdx.z;
  int outputPlane = threadIdx.x;

  float sum = 0.0f;
  // guard against horizontal tiling overhang
  if (outputCol < gradOutput.getSize(kWidthDim)) {
    for (int image = 0; image < BatchSize; ++image) {
      sum += gradOutput[image][outputRow][outputCol][outputPlane];
    }

    gradBias[outputRow][outputCol][outputPlane] += sum * scale;
  }
}

// This dispatcher method determines if an intra-warp reduction is
// possible to reduce same-warp atomicAdd() usage (i.e. collisions).
//
template <int BatchSize>
void
dispatchAccGradBiases(cudaStream_t stream,
                      DeviceTensor<float, 4> input,
                      DeviceTensor<float, 4> gradOutput,
                      DeviceTensor<float, 3> gradBias,
                      float scale,
                      int dH, int dW) {
  const int kBlockSize = 256; // threads
  int totalPlanes = gradOutput.getSize(kPlaneDim)
    * input.getSize(kPlaneDim);
  assert(gradOutput.getSize(kPlaneDim) <= 256);
  // assign subsequent threads to the output planes
  // if output planes is much smaller than kBlockSize tile horizontally
  dim3 block(gradOutput.getSize(kPlaneDim),
             kBlockSize / gradOutput.getSize(kPlaneDim));
  // tile blocks over the complete output tensor width and hight
  dim3 grid(1, grid.y = cuda::ceil(gradOutput.getSize(kWidthDim),
                                   static_cast<int>(block.y)),
            gradOutput.getSize(kHeightDim));
  accGradBiasBatch<BatchSize><<<grid, block, 0, stream>>>(
    gradOutput, gradBias, scale);
}

// Macro used in dispatcher function below.
#define ACC_GRAD_PARAMETERS_CASE(B) case B:                             \
  dispatchAccGradWeightsITP<B>(stream, input, gradOutput,               \
                               gradWeight, scale, dH, dW);              \
  dispatchAccGradBiases<B>(stream, input, gradOutput,                   \
                           gradBias, scale, dH, dW);                    \
  break

// Dispatcher function that binds the batchSize, which must be a power-of-two
// (POT) to a function template with the batch size baked in.
void accGradParametersBatchPOT(cudaStream_t stream,
                               DeviceTensor<float, 4> input,
                               DeviceTensor<float, 4> gradOutput,
                               DeviceTensor<float, 6> gradWeight,
                               DeviceTensor<float, 3> gradBias,
                               float scale,
                               int batchSize, int dH, int dW) {
  switch (batchSize) {
    ACC_GRAD_PARAMETERS_CASE(128);
    ACC_GRAD_PARAMETERS_CASE(64);
    ACC_GRAD_PARAMETERS_CASE(32);
    ACC_GRAD_PARAMETERS_CASE(16);
    ACC_GRAD_PARAMETERS_CASE(8);
    ACC_GRAD_PARAMETERS_CASE(4);
    ACC_GRAD_PARAMETERS_CASE(2);
    ACC_GRAD_PARAMETERS_CASE(1);
    default:
      assert(false); // input validation, for debugging only
  }
}

// based on perf benchmarking for K40 a batchsize of 32 reaches maximum
// kernel efficiency, which starts to drop again above this threshold (due to
// excessive register pressure/local memory spilling).
//
const int kAccGradParametersMaxBatchSize = 32;

// Breaks the problem up into batches that are powers of two.
void locallyConnectedAccGradParameters(cudaStream_t stream,
                                       const float* input,
                                       const float* gradOutput,
                                       float* gradWeight,
                                       float* gradBias,
                                       float scale,
                                       LocallyConnectedParam& params) {
  long batchIdx = 0;
  int weightSize[6] = {params.outputHeight, params.outputWidth,
    params.kernelHeight, params.kernelWidth,
    params.outputPlanes, params.inputPlanes};
  DeviceTensor<float, 6> cudaGradWeight(gradWeight, weightSize);
  int biasSize[3] = {params.outputHeight, params.outputWidth,
                     params.outputPlanes};
  DeviceTensor<float, 3> cudaGradBias(gradBias, biasSize);

  int batchSize = kAccGradParametersMaxBatchSize;
  int inputSize[4] = {batchSize, params.inputHeight, params.inputWidth,
    params.inputPlanes};
  int outputSize[4] = {batchSize, params.outputHeight, params.outputWidth,
    params.outputPlanes};

  // break problem down along batch dimesion into power-of-two sized batches
  while (batchSize > 0) {
    while (batchIdx < (params.batchSize / batchSize) * batchSize) {
      DeviceTensor<float, 4> cudaInput(const_cast<float*>(input), inputSize);
      DeviceTensor<float, 4> cudaGradOutput(const_cast<float*>(gradOutput),
                                            outputSize);
      accGradParametersBatchPOT(stream, cudaInput, cudaGradOutput,
                                cudaGradWeight, cudaGradBias,
                                scale, batchSize,
                                params.dH, params.dW);
      batchIdx += batchSize;
      input += cudaInput.numElements();
      gradOutput += cudaGradOutput.numElements();
    }
    batchSize /= 2;
    inputSize[0] = batchSize;
    outputSize[0] = batchSize;
  }
}

} // detail namespace

}}}  // namespaces

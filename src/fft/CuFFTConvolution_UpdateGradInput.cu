// Copyright 2004-present Facebook. All Rights Reserved.

#include "CuFFTConvolution_UpdateGradInput.cuh"

#include "cuda/CudaUtils.cuh"
#include "cuda/DeviceTensor.cuh"
#include "DeviceTensorUtils.h"
#include "THCTensor.h"
#include "CuBLASWrapper.h"
#include "CuFFTWrapper.cuh"
#include "CuFFTConvolution.cuh"
#include "Utils.cuh"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <glog/logging.h>

using namespace facebook::cuda;

namespace facebook { namespace deeplearning { namespace torch {

// Assumes complex is float[2]
__global__ void referenceUpdateGradInput(DeviceTensor<float, 5> inputComplex,
                                         DeviceTensor<float, 5> weightComplex,
                                         DeviceTensor<float, 5> outputComplex)
{
  // Input originally real, we have circular Hermitian symmetry:
  // X[k] = X∗[−k mod N] .
  const int Batches = inputComplex.getSize(0);
  const int Weight = weightComplex.getSize(0);
  const int InputRows = inputComplex.getSize(2);
  const int InputCols = inputComplex.getSize(3);
  for (int batch = 0; batch < Batches; ++batch) {
    for (int filter = 0; filter < Weight; ++filter) {
      for (int inputRow = 0; inputRow < InputRows; ++inputRow) {
        for (int inputCol = 0; inputCol < InputCols; ++inputCol) {
          for (int inputPlane = 0; inputPlane < inputComplex.getSize(1);
               ++inputPlane) {
            cuFloatComplex* inp = inputComplex[batch][inputPlane]
              [inputRow][inputCol].dataAs<cuFloatComplex>();
            if (filter == 0) {
              inp->x = 0.0f;
              inp->y = 0.0f;
            }

            cuFloatComplex weight = weightComplex[filter][inputPlane]
              [inputRow][inputCol].ldgAs<cuFloatComplex>();

            cuFloatComplex output = outputComplex[batch][filter][inputRow]
                      [inputCol].ldgAs<cuFloatComplex>();

            *inp = cuCfmaf(weight, output, *inp);
          }
        }
      }
    }
  }
}

void CuFFTConvolution_ReferenceUpdateGradInput(THCState* state,
                                               THCudaTensor* inputTH,
                                               THCudaTensor* weightTH,
                                               THCudaTensor* outputTH,
                                               THCudaTensor* inputComplexTH,
                                               THCudaTensor* weightComplexTH,
                                               THCudaTensor* outputComplexTH) {
  DeviceTensor<float, 4> weight =
    torchToDeviceTensor<float, 4>(state, weightTH);
  DeviceTensor<float, 4> input =
    torchToDeviceTensor<float, 4>(state, inputTH);
  DeviceTensor<float, 4> output =
    torchToDeviceTensor<float, 4>(state, outputTH);

  DeviceTensor<float, 5> inputComplex =
    torchToDeviceTensor<float, 5>(state, inputComplexTH);
  DeviceTensor<float, 5> outputComplex =
    torchToDeviceTensor<float, 5>(state, outputComplexTH);
  DeviceTensor<float, 5> weightComplex =
    torchToDeviceTensor<float, 5>(state, weightComplexTH);

  fft2d<2>(weight, weightComplex);
  fft2d<2>(output, outputComplex);

  dim3 grid(1);
  dim3 block(1);
  referenceUpdateGradInput<<<grid, block>>>(
    inputComplex, weightComplex, outputComplex);

  fft2d<2>(input, inputComplex, FFTParameters().inverse());
}

void CuFFTConvolution_UpdateGradInput(THCState* state,
                                      THCudaTensor* inputTH,
                                      THCudaTensor* weightTH,
                                      THCudaTensor* outputTH,
                                      THCudaTensor* inputComplexTH,
                                      THCudaTensor* weightComplexTH,
                                      THCudaTensor* outputComplexTH,
                                      THCudaTensor* inputComplexTTH,
                                      THCudaTensor* weightComplexTTH,
                                      THCudaTensor* outputComplexTTH) {
  CuFFTConvolution conv((ConvolutionPass(ConvolutionPass::kUpdateGradInput)));
  conv.withInputAndBuffers(state, inputTH, inputComplexTH, inputComplexTTH)
    .withFiltersAndBuffers(state, weightTH, weightComplexTH, weightComplexTTH)
    .withOutputAndBuffers(state, outputTH, outputComplexTH, outputComplexTTH)
    .run();
}

void CuFFTConvolution_UpdateGradInput(CuFFTConvolution* conv) {
  conv->run();
}

} } } // namespace

// Copyright 2004-present Facebook. All Rights Reserved.

#include "CuFFTConvolution_AccGradParameters.cuh"

#include "cuda/CudaUtils.cuh"
#include "cuda/DeviceTensor.cuh"
#include "DeviceTensorUtils.h"
#include "THCTensor.h"
#include "CuBLASWrapper.h"
#include "ConvolutionBias.cuh"
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
__global__ void
referenceAccGradParameters(DeviceTensor<float, 5> inputComplex,
                           DeviceTensor<float, 5> filtersComplex,
                           DeviceTensor<float, 5> gradOutputComplex,
                           float scale) {
  // Input originally real, we have circular Hermitian symmetry:
  // X[k] = X∗[−k mod N] .
  const int Batches = inputComplex.getSize(0);
  const int Filters = filtersComplex.getSize(0);
  const int FiltersRows = filtersComplex.getSize(2);
  const int FiltersCols = filtersComplex.getSize(3);
  for (int batch = 0; batch < Batches; ++batch) {
    for (int filter = 0; filter < Filters; ++filter) {
      for (int filtersRow = 0; filtersRow < FiltersRows; ++filtersRow) {
        for (int filtersCol = 0; filtersCol < FiltersCols; ++filtersCol) {
          for (int inputPlane = 0; inputPlane < inputComplex.getSize(1);
               ++inputPlane) {
            cuFloatComplex* filt = filtersComplex[filter][inputPlane]
              [filtersRow][filtersCol].dataAs<cuFloatComplex>();
            if (batch == 0) {
              filt->x = 0.0f;
              filt->y = 0.0f;
            }

            cuFloatComplex input = inputComplex[batch][inputPlane]
              [filtersRow][filtersCol].ldgAs<cuFloatComplex>();

            cuFloatComplex gradOutput =
              cuConjf(gradOutputComplex[batch][filter][filtersRow]
                      [filtersCol].ldgAs<cuFloatComplex>());

            input.x *= scale;
            input.y *= scale;
            *filt = cuCfmaf(input, gradOutput, *filt);
          }
        }
      }
    }
  }
}

void
CuFFTConvolution_ReferenceAccGradParameters(THCState* state,
                                            THCudaTensor* inputTH,
                                            THCudaTensor* gradWeightTH,
                                            THCudaTensor* gradOutputTH,
                                            THCudaTensor* gradBiasTH,
                                            float scale,
                                            THCudaTensor* inputComplexTH,
                                            THCudaTensor* gradWeightComplexTH,
                                            THCudaTensor* gradOutputComplexTH) {
  DeviceTensor<float, 4> filters =
    torchToDeviceTensor<float, 4>(state, gradWeightTH);
  DeviceTensor<float, 4> input =
    torchToDeviceTensor<float, 4>(state, inputTH);
  DeviceTensor<float, 4> gradOutput =
    torchToDeviceTensor<float, 4>(state, gradOutputTH);

  DeviceTensor<float, 5> inputComplex =
    torchToDeviceTensor<float, 5>(state, inputComplexTH);
  DeviceTensor<float, 5> gradOutputComplex =
    torchToDeviceTensor<float, 5>(state, gradOutputComplexTH);
  DeviceTensor<float, 5> filtersComplex =
    torchToDeviceTensor<float, 5>(state, gradWeightComplexTH);

  fft2d<2>(input, inputComplex);
  fft2d<2>(gradOutput, gradOutputComplex);

  dim3 grid(1);
  dim3 block(1);
  referenceAccGradParameters<<<grid, block>>>(
    inputComplex, filtersComplex, gradOutputComplex, scale);

  fft2d<2>(filters, filtersComplex, FFTParameters().inverse());

  bias::accGradParametersBias(state, gradOutputTH, gradBiasTH, scale);
}

void CuFFTConvolution_AccGradParameters(THCState* state,
                                        THCudaTensor* inputTH,
                                        THCudaTensor* gradWeightTH,
                                        THCudaTensor* gradOutputTH,
                                        THCudaTensor* gradBiasTH,
                                        float scale,
                                        THCudaTensor* inputComplexTH,
                                        THCudaTensor* gradWeightComplexTH,
                                        THCudaTensor* gradOutputComplexTH,
                                        THCudaTensor* inputComplexTTH,
                                        THCudaTensor* gradWeightComplexTTH,
                                        THCudaTensor* gradOutputComplexTTH) {
  CuFFTConvolution conv((ConvolutionPass(ConvolutionPass::kAccGradParameters)));
  conv.withInputAndBuffers(state, inputTH, inputComplexTH, inputComplexTTH)
    .withFiltersAndBuffers(state, gradWeightTH, gradWeightComplexTH,
                           gradWeightComplexTTH)
    .withOutputAndBuffers(state, gradOutputTH, gradOutputComplexTH,
                          gradOutputComplexTTH)
    .withScale(scale)
    .run();

  bias::accGradParametersBias(state, gradOutputTH, gradBiasTH, scale);
}

void CuFFTConvolution_AccGradParameters(THCState* state,
                                        CuFFTConvolution* conv,
                                        THCudaTensor* gradOutputTH,
                                        THCudaTensor* gradBiasTH,
                                        float scale) {
  conv->run();

  bias::accGradParametersBias(state, gradOutputTH, gradBiasTH, scale);
}

} } } // namespace

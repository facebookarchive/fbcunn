// Copyright 2004-present Facebook. All Rights Reserved.

#include "CuFFTConvolution_UpdateOutput.cuh"

#include "cuda/CudaUtils.cuh"
#include "cuda/DeviceTensor.cuh"
#include "DeviceTensorUtils.h"
#include "THCTensor.h"
#include "ConvolutionBias.cuh"
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
__global__ void referenceUpdateOuput(DeviceTensor<float, 5> inputComplex,
                                     DeviceTensor<float, 5> filtersComplex,
                                     DeviceTensor<float, 5> outputComplex)
{
  // Input originally real, we have circular Hermitian symmetry:
  // X[k] = X∗[−k mod N] .
  const int Batches = inputComplex.getSize(0);
  const int Filters = filtersComplex.getSize(0);
  const int OutputRows = outputComplex.getSize(2);
  const int OutputCols = outputComplex.getSize(3);
  for (int batch = 0; batch < Batches; ++batch) {
    for (int filter = 0; filter < Filters; ++filter) {
      for (int outputRow = 0; outputRow < OutputRows; ++outputRow) {
        for (int outputCol = 0; outputCol < OutputCols; ++outputCol) {
          cuFloatComplex* out = outputComplex[batch][filter]
            [outputRow][outputCol].dataAs<cuFloatComplex>();
          out->x = 0.0f;
          out->y = 0.0f;
          for (int inputPlane = 0; inputPlane < inputComplex.getSize(1);
               ++inputPlane) {
            cuFloatComplex input =
              inputComplex[batch][inputPlane]
              [outputRow][outputCol].ldgAs<cuFloatComplex>();

            cuFloatComplex filters =
              cuConjf(filtersComplex[filter][inputPlane]
                      [outputRow][outputCol].ldgAs<cuFloatComplex>());

            *out = cuCfmaf(input, filters, *out);
          }
        }
      }
    }
  }
}

void CuFFTConvolution_ReferenceUpdateOutput(THCState* state,
                                            THCudaTensor* inputTH,
                                            THCudaTensor* kernelsTH,
                                            THCudaTensor* outputTH,
                                            THCudaTensor* biasTH,
                                            THCudaTensor* inputComplexTH,
                                            THCudaTensor* kernelsComplexTH,
                                            THCudaTensor* outputComplexTH) {
  DeviceTensor<float, 4> filters =
    torchToDeviceTensor<float, 4>(state, kernelsTH);
  DeviceTensor<float, 4> input =
    torchToDeviceTensor<float, 4>(state, inputTH);
  DeviceTensor<float, 4> output =
    torchToDeviceTensor<float, 4>(state, outputTH);

  DeviceTensor<float, 5> inputComplex =
    torchToDeviceTensor<float, 5>(state, inputComplexTH);
  DeviceTensor<float, 5> outputComplex =
    torchToDeviceTensor<float, 5>(state, outputComplexTH);
  DeviceTensor<float, 5> filtersComplex =
    torchToDeviceTensor<float, 5>(state, kernelsComplexTH);

  fft2d<2>(input, inputComplex);
  fft2d<2>(filters, filtersComplex);

  dim3 grid(1);
  dim3 block(1);
  referenceUpdateOuput<<<grid, block>>>(
    inputComplex, filtersComplex, outputComplex);

  fft2d<2>(output, outputComplex, FFTParameters().inverse());

  bias::updateOutputBias(state, outputTH, biasTH);
}

void CuFFTConvolution_UpdateOutput(THCState* state,
                                   THCudaTensor* inputTH,
                                   THCudaTensor* kernelsTH,
                                   THCudaTensor* outputTH,
                                   THCudaTensor* biasTH,
                                   THCudaTensor* inputComplexTH,
                                   THCudaTensor* kernelsComplexTH,
                                   THCudaTensor* outputComplexTH,
                                   THCudaTensor* inputComplexTTH,
                                   THCudaTensor* kernelsComplexTTH,
                                   THCudaTensor* outputComplexTTH) {
  CuFFTConvolution conv((ConvolutionPass(ConvolutionPass::kUpdateOutput)));
  conv.withInputAndBuffers(state,
                           inputTH, inputComplexTH, inputComplexTTH)
    .withFiltersAndBuffers(state,
                           kernelsTH, kernelsComplexTH, kernelsComplexTTH)
    .withOutputAndBuffers(state,
                          outputTH, outputComplexTH, outputComplexTTH)
    .run();

  bias::updateOutputBias(state, outputTH, biasTH);
}

void CuFFTConvolution_UpdateOutput(THCState* state,
                                   CuFFTConvolution* conv,
                                   THCudaTensor* outputTH,
                                   THCudaTensor* biasTH) {
  conv->run();

  bias::updateOutputBias(state, outputTH, biasTH);
}

} } } // namespace

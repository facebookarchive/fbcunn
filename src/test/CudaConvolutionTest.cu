// Copyright 2004-present Facebook. All Rights Reserved.

#include "THC.h"
#include "deeplearning/torch/layers/nyu/fft/fft_convolution.hpp"

namespace facebook { namespace deeplearning { namespace torch { namespace test {

size_t fftGetMinBufferSize(const int batchSize, const int nInputPlanes,
                           const int nOutputPlanes,
                           int iH, int iW,
                           const int kH, const int kW,
                           bool updateGradInput) {
  return getMinBufferSize(batchSize, nInputPlanes, nOutputPlanes,
                          iH, iW, kH, kW, updateGradInput);
}

void fftUpdateOutput(const float* input, const float* weight,
                     float* output, const int batchSize,
                     const int nInputPlanes, const int nOutputPlanes,
                     const int iH, const int iW,
                     const int kH, const int kW,
                     cuComplex* memoryBuffer) {
  fft_updateOutput(input, weight, output, batchSize,
                   nInputPlanes, nOutputPlanes,
                   iH, iW, kH, kW, memoryBuffer);
}

void fftUpdateGradInput(const float* gradOutput, const float* weight,
                        float* gradInput, const int batchSize,
                        const int nInputPlanes, const int nOutputPlanes,
                        const int iH, const int iW,
                        const int kH, const int kW,
                        cuComplex* memoryBuffer) {
  fft_updateGradInput(gradOutput, weight, gradInput, batchSize,
                      nInputPlanes, nOutputPlanes,
                      iH, iW, kH, kW, memoryBuffer);
}

void fftAccGradParameters(
  const float* input, const float* gradOutput, float* gradWeight,
  const int batchSize, const int nInputPlanes, const int nOutputPlanes,
  const int iH, const int iW, const int kH, const int kW,
  cuComplex* memoryBuffer) {
  fft_accGradParameters(input, gradOutput, gradWeight, batchSize,
                        nInputPlanes, nOutputPlanes,
                        iH, iW, kH, kW, memoryBuffer);
}

} } } } // namespace

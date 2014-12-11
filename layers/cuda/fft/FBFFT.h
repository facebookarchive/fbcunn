// Copyright 2004-present Facebook. All Rights Reserved.
#pragma once

#include "THCTensor.h"
#include "torch/fb/fbcunn/layers/cuda/fft/CuFFTWrapper.cuh"

namespace facebook { namespace deeplearning { namespace torch {

template <int Batch, int Dim>
FFTParameters::ErrorCode fbfft(
  THCudaTensor* real,
  THCudaTensor* complex,
  FFTParameters params = FFTParameters().withFbfft());

} } } // namespace

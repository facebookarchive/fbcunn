// Copyright 2004-present Facebook. All Rights Reserved.
#pragma once

#include "cuda/DeviceTensor.cuh"

namespace facebook { namespace deeplearning { namespace torch {

template <int Batch>
facebook::cuda::fbfft::FBFFTParameters::ErrorCode fbfft1dHost(
  facebook::cuda::DeviceTensor<float, Batch + 1>& real,
  facebook::cuda::DeviceTensor<float, Batch + 2>& complexAsFloat,
  facebook::cuda::fbfft::FBFFTParameters params =
  facebook::cuda::fbfft::FBFFTParameters(),
  cudaStream_t s = 0);

// If calling a 2D-fft of size > 32 we need a buffer to avoid a race condition
// between reads and writes to device memory on the corner turn.
template <int Batch>
facebook::cuda::fbfft::FBFFTParameters::ErrorCode fbfft2dHost(
  facebook::cuda::DeviceTensor<float, Batch + 2>& real,
  facebook::cuda::DeviceTensor<float, Batch + 3>& complexAsFloat,
  facebook::cuda::DeviceTensor<float, Batch + 3>* bufferAsFloat,
  facebook::cuda::fbfft::FBFFTParameters params =
  facebook::cuda::fbfft::FBFFTParameters(),
  cudaStream_t s = 0);

// If calling a 2D-fft of size > 32 we need a buffer to avoid a race condition
// between reads and writes to device memory on the corner turn.
template <int Batch>
facebook::cuda::fbfft::FBFFTParameters::ErrorCode fbfft(
  THCState* state,
  THCudaTensor* real,
  THCudaTensor* complex,
  THCudaTensor* buffer = nullptr,
  facebook::cuda::fbfft::FBFFTParameters params =
  facebook::cuda::fbfft::FBFFTParameters(),
  cudaStream_t s = 0);

} } } // namespace

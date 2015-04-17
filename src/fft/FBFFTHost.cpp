// Copyright 2004-present Facebook. All Rights Reserved.

#include "THCTensor.h"
#include "cuda/Complex.cuh"
#include "cuda/ComputeCapabilities.cuh"
#include "cuda/DeviceTensor.cuh"
#include "cuda/fbfft/FBFFT.h"
#include "CuFFTWrapper.cuh"
#include "DeviceTensorUtils.h"

#include <cuda_runtime.h>
#include <glog/logging.h>

using namespace facebook::cuda;
using namespace facebook::cuda::fbfft;

namespace facebook { namespace deeplearning { namespace torch {

template <int Batch>
FBFFTParameters::ErrorCode fbfft1dHost(
    DeviceTensor<float, Batch + 1>& real,
    DeviceTensor<float, Batch + 2>& complexAsFloat,
    FBFFTParameters params,
    cudaStream_t s) {
  if (params.forwardFFT()) {
    return fbfft1D<Batch>(real, complexAsFloat, s);
  } else {
    return fbifft1D<Batch>(real, complexAsFloat, s);
  }
}

template <typename T, int Dim>
void fillStrides(const T& model, int strides[Dim]) {
  for (int i = 0; i < Dim; ++i) {
    strides[i] = model.getStride(i) / (sizeof(Complex) / sizeof(float));
  }
}

template <int Batch>
FBFFTParameters::ErrorCode fbfft2dHost(
  DeviceTensor<float, Batch + 2>& real,
  DeviceTensor<float, Batch + 3>& complexAsFloat,
  DeviceTensor<float, Batch + 3>* bufferAsFloat,
  FBFFTParameters params,
  cudaStream_t s) {

  constexpr int Dim = 2;
  int strides[Batch + Dim];
  fillStrides<DeviceTensor<float, Batch + Dim + 1>, Batch + Dim>(
    complexAsFloat, strides);
  DeviceTensor<Complex, Batch + Dim> complex(
    complexAsFloat.template dataAs<Complex>(),
    complexAsFloat.sizes(),
    strides);

  if (complex.getSize(Batch + 1) >= 64) {
    // If calling a 2D-fft of size > 32 we need a buffer to avoid a race
    // condition between reads and writes to device memory on the corner
    // turn.
    assert(bufferAsFloat);
    assert((void*)bufferAsFloat->data() != (void*)complex.data());

    DeviceTensor<float, Batch + Dim + 1> bufferAsFloatTr(
      bufferAsFloat->data(),
      bufferAsFloat->sizes(), // Reuse the first Batch + Dim elements of size
      bufferAsFloat->strides());
    std::vector<int> dims({0, 2, 1, 3});
    bufferAsFloatTr.permuteDims(dims);

    fillStrides<DeviceTensor<float, Batch + Dim + 1>, Batch + Dim>(
      (*bufferAsFloat), strides);
    DeviceTensor<Complex, Batch + Dim> buffer(
      bufferAsFloat->template dataAs<Complex>(),
      bufferAsFloat->sizes(), // Reuse the first Batch + Dim elements of size[4]
      strides);

    fillStrides<DeviceTensor<float, Batch + Dim + 1>, Batch + Dim>(
      bufferAsFloatTr, strides);
    DeviceTensor<Complex, Batch + Dim> bufferTr(
      bufferAsFloatTr.template dataAs<Complex>(),
      bufferAsFloatTr.sizes(), // Reuse the first Batch + Dim elements of size
      strides);

    FBFFTParameters::ErrorCode res;
    if (params.forwardFFT()) {
      res = fbfft2D<Batch>(real, bufferAsFloatTr, s);
    } else  {
      assert(real.getSize(0) == bufferAsFloat->getSize(0));
      assert(complex.getSize(1) ==
             numHermitian(bufferAsFloat->getSize(1)));
      assert(complex.getSize(2) == bufferAsFloat->getSize(2));
      res = fbifft2D<Batch>(complexAsFloat, *bufferAsFloat);
    }

    if (res != FBFFTParameters::Success) {
      return res;
    }

    if (params.forwardFFT()) {
      return fbfft2D<Batch>(bufferTr, complex, s);
    } else {
      return fbifft2D<Batch>(buffer, real, s);
    }
  } else {
    if (params.forwardFFT()) {
      return fbfft2D<Batch>(real, complexAsFloat, s);
    } else {
      return fbifft2D<Batch>(complex, real, s);
    }
  }

  return FBFFTParameters::UnsupportedDimension;
}


template <int Batch>
FBFFTParameters::ErrorCode fbfft1dHost(THCState* state,
                                       THCudaTensor* r,
                                       THCudaTensor* c,
                                       FBFFTParameters params,
                                       cudaStream_t s) {
  constexpr int Dim = 1;
  constexpr int BatchOne = 1;
  auto real = torchToDeviceTensorCast<float, BatchOne + Dim>(state, r);
  auto complexAsFloat =
    torchToDeviceTensorCast<float, BatchOne + Dim + 1>(state, c);
  return fbfft1dHost<BatchOne>(real, complexAsFloat, params, s);
}

template <int Batch>
FBFFTParameters::ErrorCode fbfft2dHost(THCState* state,
                                       THCudaTensor* r,
                                       THCudaTensor* c,
                                       THCudaTensor* b,
                                       FBFFTParameters params,
                                       cudaStream_t s) {
  constexpr int Dim = 2;
  constexpr int BatchOne = 1;
  auto real = torchToDeviceTensorCast<float, BatchOne + Dim>(state, r);
  auto complexAsFloat =
    torchToDeviceTensorCast<float, BatchOne + Dim + 1>(state, c);
  if (!b) {
    return fbfft2dHost<BatchOne>(real, complexAsFloat, nullptr, params, s);
  }

  auto bufferAsFloat =
    torchToDeviceTensorCast<float, BatchOne + Dim + 1>(state, b);
  return fbfft2dHost<BatchOne>(real, complexAsFloat, &bufferAsFloat, params, s);
}


template <int Batch>
FBFFTParameters::ErrorCode fbfft(THCState* state,
                                 THCudaTensor* r,
                                 THCudaTensor* c,
                                 THCudaTensor* b,
                                 FBFFTParameters params,
                                 cudaStream_t s) {
  if (THCudaTensor_nDimension(state, r) - Batch == 1) {
    return fbfft1dHost<Batch>(state, r, c, params, s);
  } else if (THCudaTensor_nDimension(state, r) - Batch == 2) {
    return fbfft2dHost<Batch>(state, r, c, b, params, s);
  }
  return FBFFTParameters::UnsupportedDimension;
}

template FBFFTParameters::ErrorCode
fbfft<1>(THCState* state,
         THCudaTensor* real,
         THCudaTensor* complex,
         THCudaTensor* buffer,
         FBFFTParameters params,
         cudaStream_t s);

template FBFFTParameters::ErrorCode
fbfft<2>(THCState* state,
         THCudaTensor* real,
         THCudaTensor* complex,
         THCudaTensor* buffer,
         FBFFTParameters params,
         cudaStream_t s);

} } } // namespace

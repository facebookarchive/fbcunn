// Copyright 2004-present Facebook. All Rights Reserved.
#pragma once

#include "cuda/DeviceTensor.cuh"
#include "cuda/fbfft/FBFFT.h"
#include "Utils.cuh"

#include <cufft.h>

namespace facebook { namespace deeplearning { namespace torch {

// Can add layout stuff later if needed
class FFTParameters {
 public:
  // Default is forward, normalized FFT.
  // Normalization occurs only in inverse FFT (by 1 / (M.N)) since CuFFT does
  // unnormalized FFTs by default
  FFTParameters() :
      version(cufft), direction_(true), normalize_(true) {}

  operator facebook::cuda::fbfft::FBFFTParameters() const {
    facebook::cuda::fbfft::FBFFTParameters res;
    res = res.normalize(normalize_);
    return (direction_) ? res.forward() : res.inverse();
  }

  FFTParameters& withCufft() {
    version = cufft;
    return *this;
  }

  FFTParameters& withFbfft() {
    version = fbfft;
    return *this;
  }

  FFTParameters& forward() {
    direction_ = true;
    return *this;
  }

  FFTParameters& inverse() {
    direction_ = false;
    return *this;
  }

  FFTParameters& normalize(bool n) {
    normalize_ = n;
    return *this;
  }

  bool forwardFFT() const { return  direction_; }
  bool inverseFFT() const { return !direction_; }
  bool normalizeFFT() const { return normalize_; }
  bool cuFFT() const { return version == cufft; }
  bool fbFFT() const { return version == fbfft; }

  template <bool Hermitian>
  std::vector<long> makeComplexTensorSizes(
      long batch, long plane, long y, long x) {
    // Until fbfft supports rectangular ffts just assert it does not
    assert(cuFFT() || y == x);
    std::vector<long> result(4);
    result[0] = batch;
    result[1] = plane;
    result[2] = (fbFFT() && Hermitian) ? numHermitian(y) : y;
    result[3] = (cuFFT() && Hermitian) ? numHermitian(x) : x;
    return result;
  }

  // Replaces cufft plans in the case of fbfft, only needed for sizes > 32.
  // For <= 32 we do everything in place.
  std::vector<long> makeTmpBufferSizes(
      long batch, long plane, long y, long x) {
    assert(fbFFT());
    // Until fbfft supports rectangular ffts just assert it does not
    assert(y == x);
    if (y <= 32) {
      std::vector<long> result;
      return result;
    }
    std::vector<long> result(4);
    result[0] = batch;
    result[1] = plane;
    if (forwardFFT()) {
      result[2] = numHermitian(y);
    } else {
      result[2] = y;
    }
    result[3] = x;
    return result;
  }

  enum FFTVersion {
    cufft = 0,
    fbfft = 1
  } version;

 private:
  bool direction_;
  bool normalize_;
};

template <int NumBatch, int RealTensorDim>
cufftHandle
makeCuFFTPlan(const cuda::DeviceTensor<float, RealTensorDim>& real,
              const cuda::DeviceTensor<float, RealTensorDim + 1>& complex,
              FFTParameters params = FFTParameters());

template <int BatchDims>
void fft1d(cuda::DeviceTensor<float, BatchDims + 1>& real,
           cuda::DeviceTensor<float, BatchDims + 2>& complex,
           FFTParameters params = FFTParameters(),
           cufftHandle* plan = NULL, // cufftHandle is unsigned int, need to
                                     // encode lack of a plan
           cudaStream_t stream = NULL);

template <int BatchDims>
void fft2d(cuda::DeviceTensor<float, BatchDims + 2>& real,
           cuda::DeviceTensor<float, BatchDims + 3>& complex,
           FFTParameters params = FFTParameters(),
           cufftHandle* plan = NULL, // cufftHandle is unsigned int, need to
                                     // encode lack of a plan
           cudaStream_t stream = NULL);

template <int BatchDims>
void fft3d(cuda::DeviceTensor<float, BatchDims + 3>& real,
           cuda::DeviceTensor<float, BatchDims + 4>& complex,
           FFTParameters params = FFTParameters(),
           cufftHandle* plan = NULL, // cufftHandle is unsigned int, need to
                                     // encode lack of a plan
           cudaStream_t stream = NULL);

template <int NumBatch, int RealTensorDim>
void fft(cuda::DeviceTensor<float, RealTensorDim>& real,
         cuda::DeviceTensor<float, RealTensorDim + 1>& complex,
         FFTParameters params = FFTParameters(),
         cufftHandle* plan = NULL, // cufftHandle is unsigned int, need to
                                   // encode lack of a plan
         cudaStream_t stream = NULL);
} } } // namespace

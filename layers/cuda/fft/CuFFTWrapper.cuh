// Copyright 2004-present Facebook. All Rights Reserved.
#pragma once

#include "cuda/DeviceTensor.cuh"

#include <cufft.h>

namespace facebook { namespace deeplearning { namespace torch {

// Can add layout stuff later if needed
class FFTParameters {
 public:
  typedef int ErrorCode;
  static const ErrorCode Success = 0;
  static const ErrorCode UnsupportedSize = 1;
  static const ErrorCode UnsupportedDimension = 2;

  // Default is forward, normalized FFT.
  // Normalization occurs only in inverse FFT (by 1 / (M.N)) since CuFFT does
  // unnormalized FFTs by default
  FFTParameters() :
      direction_(true), normalize_(true), version_(cufft) {}

  FFTParameters& withCufft() {
    version_ = cufft;
    return *this;
  }

  FFTParameters& withFbfft() {
    version_ = fbfft;
    return *this;
  }

  FFTParameters& withNyufft() {
    version_ = nyufft;
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
  bool cuFFT() const { return version_ == cufft; }
  bool fbFFT() const { return version_ == fbfft; }
  bool nyuFFT() { return version_ == nyufft; }

 private:
  bool direction_;
  bool normalize_;
  enum FFTVersion {
    cufft = 0,
    fbfft = 1,
    nyufft = 2
  } version_;
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

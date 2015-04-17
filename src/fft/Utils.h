// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include "thpp/Tensor.h"
#include "THCTensor.h"
#include "CudaTensorUtils.h"

#include <glog/logging.h>
#include <vector>

namespace facebook { namespace deeplearning { namespace torch {
enum class FFTOutputSpecification : bool { InPlace = true, OutOfPlace = false };

// Given a 4-D input tensor in (?, ?, row, col) storage mode and a common
// padding specification for Rows and Cols, creates a real and complex cuda
// tensor suitable for cuFFT.
// If the FFTOutputSpecification is InPlace then complex and real alias the same
// storage buffer.
// The real 'time' tensor has:
//   - same dimensionality as the input tensor (4)
//   - same sizes as the input tensor
//   - modified strides to accommodate padding to (commonRows, commonCols)
// The complex 'frequency' tensor has:
//   - dimensionality 5 to support AoS with S == cufftComplex == float[2]
//   - size == stride == (?, ?, NumRows, NumCols / 2 + 1) to accommodate the
//     output of cufft R2C which has only 1/2 the data due to Hermitian
//     symmetry (X[k] == X*[-k mod NumCols])
//
// Warning going to multi-GPUs: In Version 6.0 only a subset of single GPU
// functionality is supported for two GPU execution.
// http://docs.nvidia.com/cuda/cufft/index.html#ixzz39Wu2cUWp
// TODO:#4846735 extend for 1-D and 3-D FFTs
// Always dim 4 (3b+1fft, 2b+2fft, 1b+3fft) atm, extend later
//
// This method always copies the real data.
// TODO(4948477) Remove the copy when it is not needed
template <int FFTDim>
std::unique_ptr<THCudaTensor, CudaTensorDeleter>
makeCuFFTTensorReal(
  THCState* state,
  THCudaTensor* in,
  const std::vector<long>& commonDims,
  THCudaTensor* candidateCudaStorageReal = nullptr,
  FFTOutputSpecification inPlace = FFTOutputSpecification::OutOfPlace);

// Given a real tensor that is properly padded for interpolation, construct a
// complex tensor that will hold the output of the CuFFT_R2C operation.
// If in place, reuse the real THCudaTensor storage
// Otherwise, if candidateCudaStorageComplex is large enough, use it.
// Otherwise allocate a new cuda buffer.
//
// This method never copies data but will fill with 0 if allocation occurs.
template <int FFTDim>
std::unique_ptr<THCudaTensor, CudaTensorDeleter>
makeCuFFTTensorComplex(
  THCState* state,
  THCudaTensor* real,
  const std::vector<long>& commonDims,
  THCudaTensor* candidateCudaStorageComplex = nullptr,
  FFTOutputSpecification inPlace = FFTOutputSpecification::OutOfPlace);

// Given a 4-D vector containing the sizes, this allocates a full tensor of
// the specified sizes with strides matching exactly.
// If candidate storage is specified it will try to reuse the storage.
// This version does not need a model tensor but requires all dims to be
// specified a-priori.
template <int FFTDim>
std::unique_ptr<THCudaTensor, CudaTensorDeleter>
makeCuFFTTensorComplex(
  THCState* state,
  const std::vector<long>& allDims,
  THCudaTensor* candidateCudaStorageComplex = nullptr);

// Make properly sized and padded real and complex tensors on the Cuda device
// This version is wasteful and always creates new storage; used in tests
template <int FFTDim>
std::pair<std::unique_ptr<THCudaTensor, CudaTensorDeleter>,
          std::unique_ptr<THCudaTensor, CudaTensorDeleter>>
makeCuFFTTensors(
  THCState* state,
  THCudaTensor* in,
  const std::vector<long>& commonDims,
  FFTOutputSpecification inPlace = FFTOutputSpecification::OutOfPlace);

// Make properly sized and padded real and complex tensors on the Cuda device
// This version is wasteful and always creates new storage; used in tests
template <int FFTDim>
std::pair<std::unique_ptr<THCudaTensor, CudaTensorDeleter>,
          std::unique_ptr<THCudaTensor, CudaTensorDeleter>>
makeCuFFTTensors(
  THCState* state,
  thpp::Tensor<float>& in,
  const std::vector<long>& commonDims,
  FFTOutputSpecification inPlace = FFTOutputSpecification::OutOfPlace);

} } } // namespace

#include "Utils-inl.h"

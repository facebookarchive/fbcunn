// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include "Utils.cuh"
#include "THC.h"

namespace facebook { namespace deeplearning { namespace torch {

namespace {

long numFFTCols(long cols) {
  return numHermitian(cols);
}

}

template <int FFTDim>
std::unique_ptr<THCudaTensor, CudaTensorDeleter>
makeCuFFTTensorReal(
  THCState* state,
  THCudaTensor* in,
  const std::vector<long>& commonDims,
  THCudaTensor* candidateCudaStorageReal = nullptr,
  FFTOutputSpecification inPlace = FFTOutputSpecification::OutOfPlace) {
  DCHECK_EQ(FFTDim, commonDims.size());
  DCHECK_EQ(4, THCudaTensor_nDimension(state, in));
  DCHECK_LE(1, FFTDim);
  DCHECK_GE(3, FFTDim);

  std::unique_ptr<THCudaTensor, CudaTensorDeleter> real;

  // If InPlace, this must have 2 * kNumFFTCols columns
  const auto strideCols = (inPlace == FFTOutputSpecification::InPlace) ?
    2 * numFFTCols(commonDims.back()) : commonDims.back();

  auto batchOrFFTDim0 = THCudaTensor_size(state, in, 0); // always batch
  auto batchOrFFTDim1 = commonDims[0];
  auto batchOrFFTDim2 = commonDims[1];
  // Overwrite, less code, negligible perf
  if (FFTDim <= 2) {
    batchOrFFTDim1 = THCudaTensor_size(state, in, 1);
    batchOrFFTDim2 = commonDims[0];
  }
  if (FFTDim <= 1) {
    batchOrFFTDim2 = THCudaTensor_size(state, in, 2);
  }
  DCHECK_LE(THCudaTensor_size(state, in, 0), batchOrFFTDim0);
  DCHECK_LE(THCudaTensor_size(state, in, 1), batchOrFFTDim1);
  DCHECK_LE(THCudaTensor_size(state, in, 2), batchOrFFTDim2);
  DCHECK_LE(THCudaTensor_size(state, in, 3), commonDims.back());

  // The real tensor is always created, allocated and filled with the data
  // 1. Allocate with the input stride to embed in a 'large enough' zero
  // enclosing tensor that can fit (commonRows x commonCols) float[2]
  {
    // needs to fit in a float[2] for the InPlace case
    std::initializer_list<long> size =
      { batchOrFFTDim0,
        batchOrFFTDim1,
        batchOrFFTDim2,
        commonDims.back() };
    std::initializer_list<long> stride =
      { batchOrFFTDim1 * batchOrFFTDim2 * strideCols,
        batchOrFFTDim2 * strideCols,
        strideCols , // needs to fit in a float[2] for the InPlace case
        1 };

    // See D1581014, for in-place fft, we want a 'full' tensor and not merely
    // the smallest amount of storage that fits the 'size' and 'stride'.
    if (candidateCudaStorageReal == nullptr ||
        *(size.begin()) * *(stride.begin()) >
        candidateCudaStorageReal->storage->size) {
      real = makeTHCudaTensorFull(state, size, stride);
    } else {
      real = makeAliasedTHCudaTensorFull(state,
                                         candidateCudaStorageReal,
                                         size,
                                         std::vector<long>(stride));
      // Reset to 0, we are using existing garbage as buffer.
      THCudaTensor_fill(state, real.get(), 0.0f);
    }
  }

  // 2. Resize to the input size to allow proper copy
  {
    long rawSize[] = {THCudaTensor_size(state, in, 0),
                      THCudaTensor_size(state, in, 1),
                      THCudaTensor_size(state, in, 2),
                      THCudaTensor_size(state, in, 3)};
    THLongStorage *sizeTH =
      thpp::LongStorage(rawSize, rawSize + 4).moveAsTH();
    SCOPE_EXIT { THLongStorage_free(sizeTH); };
    long rawStride[] = {THCudaTensor_stride(state, real.get(), 0),
                        THCudaTensor_stride(state, real.get(), 1),
                        THCudaTensor_stride(state, real.get(), 2),
                        THCudaTensor_stride(state, real.get(), 3)};
    THLongStorage *strideTH =
      thpp::LongStorage(rawStride, rawStride + 4).moveAsTH();
    SCOPE_EXIT { THLongStorage_free(strideTH); };

    // See D1581014, for in-place fft, we can just resize since the capacity
    // is that of a full tensor as per 1.
    THCudaTensor_resize(state, real.get(), sizeTH, strideTH);
  }

  // 3. Copy using the effective data size of in.
  // This achieves padding in the time domain which is necessary to properly
  // interpolate in the Fourier space.
  THCudaTensor_copy(state, real.get(), in);

  // 4. Now that the copy to CUDA is done properly, resize as
  // (commonRows x commonCols) reals within the zero padded tensor
  long rawSize[] = { batchOrFFTDim0,
                     batchOrFFTDim1,
                     batchOrFFTDim2,
                     commonDims.back()}; // size does not always match stride
  THLongStorage *sizeTH =
    thpp::LongStorage(rawSize, rawSize + 4).moveAsTH();
  SCOPE_EXIT { THLongStorage_free(sizeTH); };
  long rawStride[] = {THCudaTensor_stride(state, real.get(), 0),
                      THCudaTensor_stride(state, real.get(), 1),
                      THCudaTensor_stride(state, real.get(), 2),
                      THCudaTensor_stride(state, real.get(), 3)};
  THLongStorage *strideTH =
    thpp::LongStorage(rawStride, rawStride + 4).moveAsTH();
  SCOPE_EXIT { THLongStorage_free(strideTH); };

  // See D1581014, for in-place fft, we can just resize since the capacity
  // is that of a full tensor as per 1.
  THCudaTensor_resize(state, real.get(), sizeTH, strideTH);

  return real;
}

template <int FFTDim>
std::unique_ptr<THCudaTensor, CudaTensorDeleter>
makeCuFFTTensorComplex(
  THCState* state,
  THCudaTensor* real,
  const std::vector<long>& commonDims,
  THCudaTensor* candidateCudaStorageComplex = nullptr,
  FFTOutputSpecification inPlace = FFTOutputSpecification::OutOfPlace) {
  DCHECK_EQ(4, THCudaTensor_nDimension(state, real));
  DCHECK_LE(1, FFTDim);
  DCHECK_GE(3, FFTDim);

  // Whether in place or not, this must always have 2 * kNumFFTCols columns
  const auto strideCols = 2 * numFFTCols(commonDims.back());
  std::unique_ptr<THCudaTensor, CudaTensorDeleter> complex;

  auto batchOrFFTDim0 = THCudaTensor_size(state, real, 0); // always batch
  auto batchOrFFTDim1 = commonDims[0];
  auto batchOrFFTDim2 = commonDims[1];
  // Overwrite, less code, negligible perf
  if (FFTDim <= 2) {
    batchOrFFTDim1 = THCudaTensor_size(state, real, 1);
    batchOrFFTDim2 = commonDims[0];
  }
  if (FFTDim <= 1) {
    batchOrFFTDim2 = THCudaTensor_size(state, real, 2);
  }

  std::initializer_list<long> size =
    {batchOrFFTDim0,
     batchOrFFTDim1,
     batchOrFFTDim2,
     strideCols / 2, // size matches stride
     2};
  std::initializer_list<long> stride =
    {batchOrFFTDim1 * batchOrFFTDim2 * strideCols,
     batchOrFFTDim2 * strideCols,
     strideCols,
     2,
     1};

  // The complex tensor can alias the same device-side memory region.
  // In any case it is sized / strided properly (AoS of dimension 5).
  if (inPlace == FFTOutputSpecification::InPlace) {
    complex = makeAliasedTHCudaTensorFull(state, real, size);
  } else {
    if (candidateCudaStorageComplex == nullptr ||
        *(size.begin()) * *(stride.begin()) >
        candidateCudaStorageComplex->storage->size) {
      complex = makeTHCudaTensorFull(state, size);
    } else {
      complex = makeAliasedTHCudaTensorFull(state,
                                            candidateCudaStorageComplex,
                                            size,
                                            std::vector<long>(stride));
    }
  }

  return complex;
}

template <int FFTDim>
std::unique_ptr<THCudaTensor, CudaTensorDeleter>
makeCuFFTTensorComplex(
  THCState* state,
  const std::vector<long>& allDims,
  THCudaTensor* candidateCudaStorageComplex = nullptr) {
  DCHECK_EQ(4, allDims.size());
  DCHECK_LE(1, FFTDim);
  DCHECK_GE(3, FFTDim);

  std::unique_ptr<THCudaTensor, CudaTensorDeleter> complex;

  std::initializer_list<long> size =
    {allDims[0],
     allDims[1],
     allDims[2],
     allDims[3], // size matches stride
     2};
  std::initializer_list<long> stride =
    {2 * allDims[1] * allDims[2] * allDims[3],
     2 * allDims[2] * allDims[3],
     2 * allDims[3],
     2,
     1};

  if (candidateCudaStorageComplex == nullptr ||
      *(size.begin()) * *(stride.begin()) >
      candidateCudaStorageComplex->storage->size) {
    complex = makeTHCudaTensorFull(state, size);
  } else {
    complex = makeAliasedTHCudaTensorFull(state,
                                          candidateCudaStorageComplex,
                                          size,
                                          std::vector<long>(stride));
  }

  return complex;
}

template <int FFTDim>
std::pair<std::unique_ptr<THCudaTensor, CudaTensorDeleter>,
          std::unique_ptr<THCudaTensor, CudaTensorDeleter>>
makeCuFFTTensors(
  THCState* state,
  THCudaTensor* in,
  const std::vector<long>& commonDims,
  FFTOutputSpecification inPlace = FFTOutputSpecification::OutOfPlace) {
  auto p1 =
    makeCuFFTTensorReal<FFTDim>(
      state, in, commonDims, nullptr, inPlace);
  auto p2 =
    makeCuFFTTensorComplex<FFTDim>(
      state, p1.get(), commonDims, nullptr, inPlace);
  return make_pair(std::move(p1), std::move(p2));
}

template <int FFTDim>
std::pair<std::unique_ptr<THCudaTensor, CudaTensorDeleter>,
          std::unique_ptr<THCudaTensor, CudaTensorDeleter>>
makeCuFFTTensors(
  THCState* state,
  thpp::Tensor<float>& in,
  const std::vector<long>& commonDims,
  FFTOutputSpecification inPlace = FFTOutputSpecification::OutOfPlace) {
  auto th = copyToCuda(state, in);
  auto res = makeCuFFTTensors<FFTDim>(state, th.get(), commonDims, inPlace);
  return make_pair(std::move(res.first), std::move(res.second));
}

} } } // namespace

// Copyright 2014-present Facebook. All Rights Reserved.

#pragma once
namespace facebook { namespace deeplearning { namespace torch {

// Depending on whether cuFFT is expected, use the Hermitian symmetry
// properties that cufft exploits on the rows.
template <typename T>
__device__ __host__ T numHermitian(T commonCols) {
  return commonCols / 2 + 1;

}

}}} // namespace

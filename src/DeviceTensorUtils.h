// Copyright 2004-present Facebook. All Rights Reserved.
#pragma once

#include "cuda/DeviceTensor.cuh"
#include "THCTensor.h"

#include <cuda_runtime.h>

namespace facebook { namespace deeplearning { namespace torch {

/// Constructs a DeviceTensor initialized from a THCudaTensor. Will
/// throw if the dimensionality does not match.
template <typename T, int Dim,
          typename IndexT, template <typename U> class PtrTraits>
cuda::DeviceTensor<T, Dim, IndexT, PtrTraits>
torchToDeviceTensor(THCState* state, THCudaTensor* t);

template <typename T, int Dim, typename IndexT>
cuda::DeviceTensor<T, Dim, IndexT, cuda::DefaultPtrTraits>
torchToDeviceTensor(THCState* state, THCudaTensor* t) {
  return torchToDeviceTensor<T, Dim, IndexT, cuda::DefaultPtrTraits>(state, t);
}

template <typename T, int Dim>
cuda::DeviceTensor<T, Dim, int, cuda::DefaultPtrTraits>
torchToDeviceTensor(THCState* state, THCudaTensor* t) {
  return torchToDeviceTensor<T, Dim, int, cuda::DefaultPtrTraits>(state, t);
}

/// Constructs a DeviceTensor initialized from a THCudaTensor by
/// upcasting or downcasting the tensor to that of a different
/// dimension.
template <typename T, int Dim,
          typename IndexT, template <typename U> class PtrTraits>
cuda::DeviceTensor<T, Dim, IndexT, PtrTraits>
torchToDeviceTensorCast(THCState* state, THCudaTensor* t);

template <typename T, int Dim, typename IndexT>
cuda::DeviceTensor<T, Dim, IndexT, cuda::DefaultPtrTraits>
torchToDeviceTensorCast(THCState* state, THCudaTensor* t) {
  return
    torchToDeviceTensorCast<T, Dim, IndexT, cuda::DefaultPtrTraits>(state, t);
}

template <typename T, int Dim>
cuda::DeviceTensor<T, Dim, int, cuda::DefaultPtrTraits>
torchToDeviceTensorCast(THCState* state, THCudaTensor* t) {
  return
    torchToDeviceTensorCast<T, Dim, int, cuda::DefaultPtrTraits>(state, t);
}

} } }  // namespace

#include "DeviceTensorUtils-inl.h"

// Copyright 2004-present Facebook. All Rights Reserved.
#pragma once

#include "cuda/DeviceTensor.cuh"
#include "THCTensor.h"

#include <cuda_runtime.h>

namespace facebook { namespace deeplearning { namespace torch {

/// Constructs a DeviceTensor initialized from a THCudaTensor. Will
/// throw if the dimensionality does not match.
template <typename T, int Dim>
cuda::DeviceTensor<T, Dim> torchToDeviceTensor(THCudaTensor* t);

/// Constructs a DeviceTensor initialized from a THCudaTensor by
/// upcasting or downcasting the tensor to that of a different
/// dimension.
template <typename T, int Dim>
cuda::DeviceTensor<T, Dim> torchToDeviceTensorCast(THCudaTensor* t);

} } }  // namespace

#include "DeviceTensorUtils-inl.h"

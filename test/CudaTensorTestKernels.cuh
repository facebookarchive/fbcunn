// Copyright 2004-present Facebook. All Rights Reserved.
#pragma once

struct THCudaTensor;

///
/// Collection of kernels for testing DeviceTensor<>
///

namespace facebook { namespace deeplearning { namespace torch {

/// Assign values to the tensor via CudaTensor based on position
bool testAssignment1d(THCudaTensor* tensor);
bool testAssignment3d(THCudaTensor* tensor);

/// Test upcasting to a higher-dimensional tensor
bool testUpcast(THCudaTensor* tensor);

/// Downcast tests
bool testDowncastTo2d(THCudaTensor* tensor);
bool testDowncastTo1d(THCudaTensor* tensor);
bool testDowncastWrites(THCudaTensor* tensor);

} } } // namespace

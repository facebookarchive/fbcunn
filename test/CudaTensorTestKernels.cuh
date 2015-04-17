// Copyright 2004-present Facebook. All Rights Reserved.
#pragma once

struct THCudaTensor;

///
/// Collection of kernels for testing DeviceTensor<>
///

namespace facebook { namespace deeplearning { namespace torch {

/// Assign values to the tensor via CudaTensor based on position
bool testAssignment1d(THCState* state, THCudaTensor* tensor);
bool testAssignment3d(THCState* state, THCudaTensor* tensor);

/// Test upcasting to a higher-dimensional tensor
bool testUpcast(THCState* state, THCudaTensor* tensor);

/// Downcast tests
bool testDowncastTo2d(THCState* state, THCudaTensor* tensor);
bool testDowncastTo1d(THCState* state, THCudaTensor* tensor);
bool testDowncastWrites(THCState* state, THCudaTensor* tensor);

} } } // namespace

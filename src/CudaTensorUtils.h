// Copyright 2004-present Facebook. All Rights Reserved.
#pragma once

#include "THCTensor.h"
#include "folly/Optional.h"
#include "thpp/Tensor.h"

#include <cuda_runtime.h>
#include <memory>
#include <vector>

struct THCState;

// unique_ptr destructor for THCudaTensor
struct THCudaTensor;
struct CudaTensorDeleter {
  explicit CudaTensorDeleter(THCState* s) : state(s) {}
  CudaTensorDeleter() : state(nullptr) {}

  void operator()(THCudaTensor*);
  THCState* state;
};

// unique_ptr destructor for device-malloc'd memory
struct CudaDeleter {
  void operator()(void* p) {
    if (p) {
      cudaFree(p);
    }
  }
};

namespace facebook { namespace deeplearning { namespace torch {

/// Constructs a new THCudaTensor initialized to 0 with the given
/// sizes and strides.
/// See D1581014, this method allocates a full tensor whose storage capacity is
/// greater than strictly requested by torch.
std::unique_ptr<THCudaTensor, CudaTensorDeleter>
makeTHCudaTensorFull(THCState* state,
                     const std::vector<long>& sizes,
                     const folly::Optional<std::vector<long>>& strides =
                     folly::none);

/// Constructs a new THCudaTensor which is a view of the aliased
/// THCudaTensor with the given sizes and strides.
/// The requested size (strides(0) * sizes(0)) must fit within the input
/// Tensor otherwise overflows would occur.
std::unique_ptr<THCudaTensor, CudaTensorDeleter>
makeAliasedTHCudaTensorFull(THCState* state,
                            THCudaTensor* in,
                            const std::vector<long>& sizes,
                            const folly::Optional<std::vector<long>>& strides =
                            folly::none);

/// See D1581014, this method allocates a full tensor whose storage capacity is
/// greater than strictly requested by torch.
std::unique_ptr<THCudaTensor, CudaTensorDeleter>
makeTHCudaTensorFull(THCState* state,
                     std::initializer_list<long> sizes,
                     std::initializer_list<long> strides =
                     std::initializer_list<long>());

/// Copy a THCudaTensor to a new host-resident Tensor. Does not modify 'tensor'.
thpp::Tensor<float> copyFromCuda(THCState* state,
                                 const THCudaTensor* tensor);

/// Copy a Tensor<float> to a new THCudaTensor. Does not modify 'tensor'.
std::unique_ptr<THCudaTensor, CudaTensorDeleter>
copyToCuda(THCState* state, thpp::Tensor<float>& tensor);

template <typename T>
std::unique_ptr<T, CudaDeleter> cudaAlloc(size_t size) {
  T* ptr = nullptr;
  const auto err = cudaMalloc(&ptr, size);

  if (!ptr || err == cudaErrorMemoryAllocation) {
    throw std::bad_alloc();
  }

  return std::unique_ptr<T, CudaDeleter>(ptr, CudaDeleter());
}

} } }  // namespace

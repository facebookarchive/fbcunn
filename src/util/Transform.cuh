// Copyright 2004-, Facebook, Inc. All Rights Reserved.

#pragma once

#include <stdint.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace facebook { namespace CUDAUtil {

/*
 * A generic interface for dense point-to-point operations.
 */
template<typename Operator>
void transform(cudaStream_t stream,
               const typename Operator::Input* input,
               typename Operator::Output* out, size_t n);

typedef uint16_t half_t;

// Some pointwise operations. They must publicly define Input and
// Output types, and provide an operator() mapping one input to one
// output.
struct ToHalf {
  typedef float Input;
  typedef half_t Output;
  Output __device__ operator()(const Input f) {
    return __float2half_rn(f);
  }
};

struct ToFloat {
  typedef half_t Input;
  typedef float Output;
  Output __device__ operator()(const Input h) {
    return __half2float(h);
  }
};

} }

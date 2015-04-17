// Copyright 2004-, Facebook, Inc. All Rights Reserved.

#include <algorithm>
#include <assert.h>

#include "util/Transform.cuh"

namespace facebook { namespace CUDAUtil {

template<typename Operator>
__global__ static void
transformKernel(const typename Operator::Input* input,
                typename Operator::Output* out,
                size_t n) {

  Operator op;
  size_t start = threadIdx.x + blockIdx.x * blockDim.x;
  if (start >= n) return;
  out[start] = op(input[start]);
}

size_t roundUp(double d) {
  return size_t(ceil(d));
}

template<typename Op>
void transform(cudaStream_t stream,
               const typename Op::Input* input,
               typename Op::Output* out,
               size_t n) {
  static const int kThreadsPerBlock = 128;
  assert(n > 0);
  int totalNumBlocks = int(ceil(1.0 * n / kThreadsPerBlock));
  dim3 blockDim(kThreadsPerBlock);
  dim3 gridDim(totalNumBlocks);
  transformKernel<Op><<<gridDim, blockDim, 0, stream>>>(input, out, n);
}

template void transform<ToHalf>(cudaStream_t stream,
                                const ToHalf::Input* in,
                                ToHalf::Output* out,
                                size_t n);
template void transform<ToFloat>(cudaStream_t stream,
                                 const ToFloat::Input* in,
                                 ToFloat::Output* out,
                                 size_t n);

} }

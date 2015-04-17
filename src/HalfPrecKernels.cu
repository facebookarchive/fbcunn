// Copyright 2004-, Facebook, Inc. All Rights Reserved.

#include <stdio.h>
#include <stdexcept>
#include <cuda.h>

#include "HalfPrec.h"
#include "util/Transform.cuh"

using namespace facebook::CUDAUtil;
void halfprec_ToHalf(cudaStream_t stream,
                     const float* input,
                     half_t* output,
                     size_t n) {
  transform<ToHalf>(stream, input, output, n);
}

void halfprec_ToFloat(cudaStream_t stream,
                      const half_t* input,
                      float* output,
                      size_t n) {
  transform<ToFloat>(stream, input, output, n);
}

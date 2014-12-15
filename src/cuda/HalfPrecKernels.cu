// Copyright 2004-, Facebook, Inc. All Rights Reserved.

#include <stdio.h>
#include <stdexcept>
#include <cuda.h>

#include "HalfPrec.h"
#include "util/Transform.cuh"

using namespace facebook::CUDAUtil;
void halfprec_ToHalf(const float* input,
                     half_t* output,
                     size_t n) {
  transform<ToHalf>(input, output, n);
}

void halfprec_ToFloat(const half_t* input,
                      float* output,
                      size_t n) {
  transform<ToFloat>(input, output, n);
}


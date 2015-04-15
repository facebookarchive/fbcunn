// Copyright 2004-, Facebook, Inc. All Rights Reserved.

#pragma once
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdlib.h>

typedef uint16_t half_t;

void halfprec_ToHalf(cudaStream_t stream,
                     const float* input,
                     half_t* output,
                     size_t n);

void halfprec_ToFloat(cudaStream_t stream,
                      const half_t* input,
                      float* output,
                      size_t n);

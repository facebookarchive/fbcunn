// Copyright 2004-, Facebook, Inc. All Rights Reserved.

#pragma once
#include <stdint.h>
#include <stdlib.h>

typedef uint16_t half_t;

void halfprec_ToHalf(const float* input,
                     half_t* output,
                     size_t n);

void halfprec_ToFloat(const half_t* input,
                      float* output,
                      size_t n);

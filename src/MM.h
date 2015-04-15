// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include "cuda/DeviceTensor.cuh"

#include <cuda_runtime.h>

namespace facebook { namespace deeplearning { namespace torch {

template <int Dim, bool ConjugateTransposeA, bool ConjugateTransposeB>
void transposeMM(facebook::cuda::DeviceTensor<float, Dim>& A,
                 facebook::cuda::DeviceTensor<float, Dim>& B,
                 facebook::cuda::DeviceTensor<float, Dim>& C,
                 float invNorm,
                 cudaStream_t s = 0);

}}}

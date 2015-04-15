// Copyright 2004-present Facebook. All Rights Reserved.

#include "cuda/DeviceTensor.cuh"
#include "cuda/MM.cuh"

using namespace facebook::cuda;

namespace facebook { namespace deeplearning { namespace torch {

template <int Dim, bool ConjugateTransposeA, bool ConjugateTransposeB>
void transposeMM(DeviceTensor<float, Dim>& A,
                 DeviceTensor<float, Dim>& B,
                 DeviceTensor<float, Dim>& C,
                 float invNorm,
                 cudaStream_t s = 0) {
  facebook::cuda::transposeMM<Dim, ConjugateTransposeA, ConjugateTransposeB>(
    A, B, C, invNorm, s);
}

#define INSTANTIATE_TRANSPOSE_MM(DIM, CONJA, CONJB)     \
  template void transposeMM<DIM, CONJA, CONJB>(         \
    DeviceTensor<float, DIM>& A,                        \
    DeviceTensor<float, DIM>& B,                        \
    DeviceTensor<float, DIM>& C,                        \
    float invNorm,                                      \
    cudaStream_t s);

INSTANTIATE_TRANSPOSE_MM(5, true, false);
INSTANTIATE_TRANSPOSE_MM(5, false, true);
INSTANTIATE_TRANSPOSE_MM(5, false, false);

#undef INSTANTIATE_TRANSPOSE_MM

}}}

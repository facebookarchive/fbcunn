// Copyright 2004-present Facebook. All Rights Reserved.

#include "DeviceTensorUtils.h"
#include "THCTensor.h"

#include "cuda/DeviceTensor.cuh"
#include "cuda/MM.cuh"


using namespace facebook::cuda;

namespace facebook { namespace deeplearning { namespace torch {

template
<int Dim, bool ConjugateTransposeA, bool ConjugateTransposeB, bool Accumulate>
void transposeMM(DeviceTensor<float, Dim>& A,
                 DeviceTensor<float, Dim>& B,
                 DeviceTensor<float, Dim>& C,
                 float invNorm,
                 cudaStream_t s = 0) {
  facebook::cuda::transposeMM
    <Dim, ConjugateTransposeA, ConjugateTransposeB, Accumulate>(
      A, B, C, invNorm, s);
}

#define INSTANTIATE_TRANSPOSE_MM(DIM, CONJA, CONJB, ACC)        \
  template void transposeMM<DIM, CONJA, CONJB, ACC>(            \
    DeviceTensor<float, DIM>& A,                                \
    DeviceTensor<float, DIM>& B,                                \
    DeviceTensor<float, DIM>& C,                                \
    float invNorm,                                              \
    cudaStream_t s);

INSTANTIATE_TRANSPOSE_MM(5, true, false, true);
INSTANTIATE_TRANSPOSE_MM(5, false, true, true);
INSTANTIATE_TRANSPOSE_MM(5, false, false, true);
INSTANTIATE_TRANSPOSE_MM(5, true, false, false);
INSTANTIATE_TRANSPOSE_MM(5, false, true, false);
INSTANTIATE_TRANSPOSE_MM(5, false, false, false);

#define CALL_TRANSPOSE_MM(DIM, CONJA, CONJB, ACC)                            \
  if (THCudaTensor_nDimension(state, tA) == DIM &&                           \
      conjugateTransposeA == CONJA &&                                        \
      conjugateTransposeB == CONJB &&                                        \
      accumulate == ACC) {                                                   \
    DeviceTensor<float, DIM> A = torchToDeviceTensor<float, DIM>(state, tA); \
    DeviceTensor<float, DIM> B = torchToDeviceTensor<float, DIM>(state, tB); \
    DeviceTensor<float, DIM> C = torchToDeviceTensor<float, DIM>(state, tC); \
    facebook::deeplearning::torch::transposeMM<DIM, CONJA, CONJB, ACC>(      \
      A, B, C, invNorm, THCState_getCurrentStream(state));                   \
    return;                                                                  \
  }

extern "C" void transposeMMFFI(THCState* state,
                               THCudaTensor* tA,
                               THCudaTensor* tB,
                               THCudaTensor* tC,
                               float invNorm,
                               bool conjugateTransposeA,
                               bool conjugateTransposeB,
                               bool accumulate) {
  CHECK_EQ(THCudaTensor_nDimension(state, tA),
           THCudaTensor_nDimension(state, tB));
  CHECK_EQ(THCudaTensor_nDimension(state, tA),
           THCudaTensor_nDimension(state, tC));

  CALL_TRANSPOSE_MM(5, true, false, true);
  CALL_TRANSPOSE_MM(5, false, true, true);
  CALL_TRANSPOSE_MM(5, false, false, true);
  CALL_TRANSPOSE_MM(5, true, false, false);
  CALL_TRANSPOSE_MM(5, false, true, false);
  CALL_TRANSPOSE_MM(5, false, false, false);
}

#undef INSTANTIATE_TRANSPOSE_MM

}}}

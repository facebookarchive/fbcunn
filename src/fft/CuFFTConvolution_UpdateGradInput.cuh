// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

struct THCudaTensor;
struct THCState;

namespace facebook { namespace deeplearning { namespace torch {

void CuFFTConvolution_ReferenceUpdateGradInput(THCState* state,
                                               THCudaTensor* inputTH,
                                               THCudaTensor* kernelsTH,
                                               THCudaTensor* outputTH,
                                               THCudaTensor* inputComplexTH,
                                               THCudaTensor* kernelsComplexTH,
                                               THCudaTensor* outputComplexTH);

// CuFFTConvolution calls require 2 sets of buffers for each
// input / kernels / output tensor.
// - The first set is used to perform FFTs
// - The second set is used to hold the transpose of the FFTs for the
//   subsequent gemm calls.
// The first set must always be supplied, the second will be constructed if
// passed NULL.
void CuFFTConvolution_UpdateGradInput(THCState* state,
                                      THCudaTensor* inputTH,
                                      THCudaTensor* kernelsTH,
                                      THCudaTensor* outputTH,
                                      THCudaTensor* inputComplexTH,
                                      THCudaTensor* kernelsComplexTH,
                                      THCudaTensor* outputComplexTH,
                                      THCudaTensor* inputComplexTTH,
                                      THCudaTensor* kernelsComplexTTH,
                                      THCudaTensor* outputComplexTTH);

class CuFFTConvolution;

// This version can be preconfigured with cublasHandle, cufftHandle and
// cudaStreams. Use this one for performance and reuse resources.
void CuFFTConvolution_UpdateGradInput(CuFFTConvolution* conv);

} } } // namespace

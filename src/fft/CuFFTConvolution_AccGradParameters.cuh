// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

struct THCudaTensor;
struct THCState;

namespace facebook { namespace deeplearning { namespace torch {

void CuFFTConvolution_ReferenceAccGradParameters(THCState* state,
                                                 THCudaTensor* inputTH,
                                                 THCudaTensor* kernelsTH,
                                                 THCudaTensor* outputTH,
                                                 THCudaTensor* gradBiasTH,
                                                 float scale,
                                                 THCudaTensor* inputComplexTH,
                                                 THCudaTensor* kernelsComplexTH,
                                                 THCudaTensor* outputComplexTH);

void CuFFTConvolution_AccGradParameters(THCState* state,
                                        THCudaTensor* inputTH,
                                        THCudaTensor* kernelsTH,
                                        THCudaTensor* outputTH,
                                        THCudaTensor* gradBiasTH,
                                        float scale,
                                        THCudaTensor* inputComplexTH,
                                        THCudaTensor* kernelsComplexTH,
                                        THCudaTensor* outputComplexTH,
                                        THCudaTensor* inputComplexTTH,
                                        THCudaTensor* kernelsComplexTTH,
                                        THCudaTensor* outputComplexTTH);

class CuFFTConvolution;


// This version can be preconfigured with cublasHandle, cufftHandle and
// cudaStreams. Use this one for performance and reuse resources.
void CuFFTConvolution_AccGradParameters(THCState* state,
                                        CuFFTConvolution* conv,
                                        THCudaTensor* gradOutputTH,
                                        THCudaTensor* gradBiasTH,
                                        float scale);
} } } // namespace

// Copyright 2004-present Facebook. All Rights Reserved.
#pragma once

namespace facebook { namespace deeplearning { namespace torch { namespace test {

// CUDA convolution layer code exposed for testing

size_t fftGetMinBufferSize(
  const int batchSize, const int nInputPlanes, const int nOutputPlanes,
  int iH, int iW, const int kH, const int kW,
  bool updateGradInput = false);

void fftUpdateOutput(
  const float* input, const float* weight, float* output,
  const int batchSize, const int nInputPlanes, const int nOutputPlanes,
  const int iH, const int iW, const int kH, const int kW,
  cuComplex* memoryBuffer);

void fftUpdateGradInput(
  const float* gradOutput, const float* weight, float* gradInput,
  const int batchSize, const int nInputPlanes, const int nOutputPlanes,
  const int iH, const int iW, const int kH, const int kW,
  cuComplex* memoryBuffer);

void fftAccGradParameters(
  const float* input, const float* gradOutput, float* gradWeight,
  const int batchSize, const int nInputPlanes, const int nOutputPlanes,
  const int iH, const int iW, const int kH, const int kW,
  cuComplex* memoryBuffer);

} } } }

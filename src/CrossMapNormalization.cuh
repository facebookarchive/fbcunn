/**
 * Copyright 2014 Facebook
 * @author Tudor Bosman (tudorb@fb.com)
 */
#pragma once

namespace facebook { namespace deeplearning { namespace torch {
namespace detail {

struct CrossMapNormalizationParam {
  int batchSize;
  int numFeatures;
  int featureSize;
  int kernelSize;
  int kernelRadius;
  float scale;
  float power;
};

void launchCrossMapNormalizationUpdateOutputKernel(
  cudaStream_t stream,
  const float* input,
  float* output,
  float* squaredSum,
  CrossMapNormalizationParam params);

void launchCrossMapNormalizationUpdateGradInputKernel(
  cudaStream_t stream,
  const float* input,
  const float* gradOutput,
  const float* squaredSum,
  float* gradInput,
  CrossMapNormalizationParam params);

}}}}  // namespaces

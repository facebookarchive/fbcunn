// Copyright 2004-present Facebook. All Rights Reserved.

#include "cuda/DeviceTensor.cuh"

namespace facebook { namespace deeplearning { namespace torch {
namespace detail {

void runSparseNLLCriterion_updateOutput(
  cudaStream_t stream,
  const cuda::DeviceTensor<float, 2>& targetIdx,
  const cuda::DeviceTensor<float, 2>& targetP,
  const cuda::DeviceTensor<float, 2>& input,
  cuda::DeviceTensor<float, 1>& output);

void runSparseNLLCriterion_updateGradInput(
  cudaStream_t stream,
  const cuda::DeviceTensor<float, 2>& targetIdx,
  const cuda::DeviceTensor<float, 2>& targetP,
  cuda::DeviceTensor<float, 2>& gradinput);

}}}}

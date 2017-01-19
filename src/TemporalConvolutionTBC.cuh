// Copyright 2004-present Facebook. All Rights Reserved.

#include "THCGeneral.h"
#include "cuda/DeviceTensor.cuh"

namespace facebook {
namespace deeplearning {
namespace torch {
namespace detail {

void runTemporalConvolutionTBC_updateOutput(
    THCState* state,
    const cuda::DeviceTensor<float, 3>& input,
    const cuda::DeviceTensor<float, 3>& output,
    const cuda::DeviceTensor<float, 3>& weight,
    const cuda::DeviceTensor<float, 1>& bias);

void runTemporalConvolutionTBC_updateGradInput(
    THCState* state,
    const cuda::DeviceTensor<float, 3>& dInput,
    const cuda::DeviceTensor<float, 3>& dOutput,
    const cuda::DeviceTensor<float, 3>& weight);

void runTemporalConvolutionTBC_accGradParameters(
    THCState* state,
    const cuda::DeviceTensor<float, 3>& input,
    const cuda::DeviceTensor<float, 3>& dOutput,
    const cuda::DeviceTensor<float, 3>& dWeight,
    const cuda::DeviceTensor<float, 1>& dBias,
    float scale);
}
}
}
}

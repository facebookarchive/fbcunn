// Copyright 2004-present Facebook. All Rights Reserved.

#include "cuda/DeviceTensor.cuh"

namespace facebook { namespace deeplearning { namespace torch {

void
runQuantize1Bit(cudaStream_t stream,
                const cuda::DeviceTensor<float, 2>& in,
                cuda::DeviceTensor<float, 2>& out,
                cuda::DeviceTensor<float, 2>& quantizationError,
                cuda::DeviceTensor<float, 1>& avgPos,
                cuda::DeviceTensor<float, 1>& avgNeg);

void
runDequantize1Bit(cudaStream_t stream,
                  const cuda::DeviceTensor<float, 2>& in,
                  const cuda::DeviceTensor<float, 1>& avgPos,
                  const cuda::DeviceTensor<float, 1>& avgNeg,
                  cuda::DeviceTensor<float, 2>& out);

} } }

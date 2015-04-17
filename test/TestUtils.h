// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once
#include "cuda/DeviceTensor.cuh"
#include "torch/fb/fbcunn/src/CudaTensorUtils.h"
#include "torch/fb/fbcunn/src/DeviceTensorUtils.h"
#include "THCTensor.h"
#include "torch/fb/fbcunn/src/fft/CuFFTConvolution_UpdateOutput.cuh"
#include "torch/fb/fbcunn/src/fft/Utils.h"

#include <folly/Optional.h>
#include <tuple>

namespace facebook { namespace deeplearning { namespace torch { namespace test {

// Constructs a full CUDA tensor of the same size as the input
std::unique_ptr<THCudaTensor, CudaTensorDeleter>
makeTHCudaTensorSameSize(THCState* state, const thpp::Tensor<float>& t);

// Constructs a full CUDA tensor with constant values
thpp::Tensor<float>
makeRandomTestTensor(std::initializer_list<long> sizeList);

thpp::Tensor<float> makeTestTensor(std::initializer_list<long> sizeList,
                                   float constant);

// Constructs a CUDA tensor by scaling the factor list
thpp::Tensor<float> makeTestTensor(
  std::initializer_list<long> sizeList,
  std::initializer_list<float> factorList,
  const folly::Optional<std::tuple<long, long, long, long>>& padding =
  folly::none);

// Constructs a full CUDA tensor by scaling {0.1f, 0.2f, 0.3f, 0.4f}
thpp::Tensor<float> makeTestTensor(std::initializer_list<long> sizeList);


bool isWithin(float a, float b, float relativeError = 1e-5f);

// Returns true or false if the two tensors match within some relative
// error; also returns the 2d slice where they first differ as a
// string if they do.
// PrecisionDebug controls how many digits are printed on error in the
// returned string.
// If compareInter is set to true, comparison will only be performed on the
// intersection subtensors:
// [0, min(reference.size(0), test.size(0))] x ... x
//   [0, min(reference.size(dim-1), test.size(dim-1))]
// This is useful for kernels that write tail garbage
std::pair<bool, std::string>
compareTensors(const thpp::Tensor<float>& reference,
               const thpp::Tensor<float>& test,
               float relativeError = 1e-5f,
               int precisionDebug = 4,
               bool compareInter = false);

// Constructs a full CUDA tensor of the same size as the input
template <int Dim>
std::unique_ptr<THCudaTensor, CudaTensorDeleter>
makeTHCudaTensorSameSize(THCState* state,
                         const cuda::DeviceTensor<float, Dim>& t) {
  std::vector<long> sizes;
  std::vector<long> strides;
  for (int i = 0; i < Dim; ++i) {
    sizes.push_back(t.getSize(i));
    strides.push_back(t.getStride(i));
  }

  return makeTHCudaTensorFull(state, sizes, strides);
}

}}}} // namespace

// Copyright 2004-present Facebook. All Rights Reserved.
#pragma once

#include "torch/fb/fbcunn/src/Tensor.h"

#include <folly/Optional.h>
#include <tuple>

namespace facebook { namespace deeplearning { namespace torch { namespace test {

///
/// Reference convolution/cross-correlation implementations
///

/// Returns the output size based on the input and filter size and
/// stride for a valid-only convolution or cross-correlation
constexpr long
getValidConvSize(long inputSize, long filterSize, long filterStride) {
  return ((inputSize - filterSize) / filterStride) + 1;
}

/// Returns the output size based on the input and filter size and
/// stride for a reverse valid-only convolution or cross-correlation
constexpr long
getValidRevConvSize(long inputSize, long filterSize, long filterStride) {
  return inputSize - (filterSize - 1) * filterStride;
}

/// Returns the output size based on the input and filter size and
/// stride for a full convolution or cross-correlation
constexpr long
getFullConvSize(long inputSize, long filterSize, long filterStride) {
  return (inputSize - 1) * filterStride + filterSize;
}

/// Input to output:
///
/// input (batch x img planes x img row x img col)
/// star (valid only)
/// filters (filter planes x img planes x filter row x filter col)
/// =
/// output (batch x filter planes x
///         getValidConvSize(img row, filter row, stride),
///         getValidConvSize(img col, filter col, stride))
/// Optional input padding is expressed as <top, bottom, left, right>
/// on each innermost 2d plane.
Tensor<float>
crossCorrelationValidOnly(
  const Tensor<float>& input,
  const Tensor<float>& filters,
  long filterRowStride,
  long filterColStride,
  const folly::Optional<std::tuple<long, long, long, long>>& padding =
  folly::none);

Tensor<float>
crossCorrelationValidOnlyInputCentric(
  const Tensor<float>& input,
  const Tensor<float>& filters,
  long filterRowStride,
  long filterColStride,
  const folly::Optional<std::tuple<long, long, long, long>>& padding =
  folly::none);

/// Output gradient to input gradient:
///
/// output (batch x filter planes x
///         getValidConvSize(img row, filter row, stride),
///         getValidConvSize(img col, filter col, stride))
/// * (full)
/// filters (filter planes x img planes x filter row x filter col)
/// =
/// input (batch x img planes x img row x img col)
/// Optional input padding is expressed as <top, bottom, left, right>
/// on each innermost 2d plane.
Tensor<float>
convolutionFull(
  const Tensor<float>& output,
  const Tensor<float>& filters,
  long filterRowStride,
  long filterColStride,
  const folly::Optional<std::tuple<long, long, long, long>>& padding =
  folly::none);

/// Output gradient to weights:
///
/// input (batch x img planes x img row x img col)
/// star (valid only)
/// output (batch x filter planes x
///         getValidRevConvSize(img row, filter row, stride),
///         getValidRevConvSize(img col, filter col, stride))
/// =
/// weight gradient (filter planes x img planes x filter row x filter col)
/// Optional input padding is expressed as <top, bottom, left, right>
/// on each innermost 2d plane. Scale is a multiplicative factor
/// applied pointwise to every output point
Tensor<float>
crossCorrelationReverseValidOnly(
  const Tensor<float>& input,
  const Tensor<float>& output,
  long filterRowStride,
  long filterColStride,
  float scale,
  const folly::Optional<std::tuple<long, long, long, long>>& padding =
  folly::none);

} } } } // namespace

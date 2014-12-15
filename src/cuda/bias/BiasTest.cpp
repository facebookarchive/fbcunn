// Copyright 2004-present Facebook. All Rights Reserved.

#include "DeviceTensorUtils.h"
#include "THCTensor.h"
#include "bias/ConvolutionBias.cuh"
#include "torch/fb/fbcunn/layers/test/TestUtils.h"

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <random>

using namespace std;
using namespace facebook::deeplearning::torch;

namespace facebook { namespace deeplearning { namespace torch { namespace bias {

constexpr int kRuns = 10;

namespace {

// Construct a feature map tensor; bias code should work on
// non-contiguous tensors, so we may create a non-contiguous layout
// for the tensor
Tensor<float>
makeOutputTensor(long batches, long planes, long rows, long cols) {
  random_device dev;
  mt19937 gen(dev());

  Tensor<float> output;
  vector<long> sizes{batches, planes, rows, cols};

  int transpose1 = uniform_int_distribution<int>(0, sizes.size() - 1)(gen);
  int transpose2 = uniform_int_distribution<int>(0, sizes.size() - 1)(gen);
  if (transpose1 != transpose2) {
    LOG(INFO) << "transposing " << transpose1 << ", " << transpose2;

    std::swap(sizes[transpose1], sizes[transpose2]);

    // the memory layout is out of order
    output =
      test::makeRandomTestTensor({sizes[0], sizes[1], sizes[2], sizes[3]});

    // this will transpose the dimensions back to normal, but the
    // tensor will be non-contiguous
    output.transpose(transpose1, transpose2);
  } else {
    // This test will be with a contiguous tensor
    output =
      test::makeRandomTestTensor({sizes[0], sizes[1], sizes[2], sizes[3]});
  }

  return output;
}

Tensor<float>
makeBiasTensor(long planes) {
  auto out = Tensor<float>({planes});

  for (long i = 0; i < planes; ++i) {
    out.at({i}) = (float) i;
  }

  return out;
}

// Reference forward bias pass
Tensor<float>
referenceBiasUpdateOutput(const Tensor<float>& output,
                          const Tensor<float>& bias) {
  auto result = Tensor<float>();
  result.resizeAs(output);
  result.copy(output);

  for (long batch = 0; batch < output.size(0); ++batch) {
    for (long plane = 0; plane < output.size(1); ++plane) {
      float biasVal = bias.at({plane});

      for (long row = 0; row < output.size(2); ++row) {
        for (long col = 0; col < output.size(3); ++col) {
          result.at({batch, plane, row, col}) += biasVal;
        }
      }
    }
  }

  return result;
}

// Reference accGrad bias pass
Tensor<float>
referenceBiasAccGradParameters(const Tensor<float>& output,
                               const Tensor<float>& bias,
                               float biasScale) {
  auto result = Tensor<float>();
  result.resizeAs(bias);
  result.copy(bias);

  for (long plane = 0; plane < output.size(1); ++plane) {
    float sum = 0.0f;

    for (long batch = 0; batch < output.size(0); ++batch) {
      for (long row = 0; row < output.size(2); ++row) {
        for (long col = 0; col < output.size(3); ++col) {
          sum += output.at({batch, plane, row, col});
        }
      }
    }

    result.at({plane}) = sum * biasScale;
  }

  return result;
}

}

TEST(BiasTest, testUpdateOutput) {
  random_device dev;
  mt19937 gen(dev());

  for (int run = 0; run < kRuns; ++run) {
    auto batchSize = uniform_int_distribution<long>(5, 15)(gen);
    auto numPlanes = uniform_int_distribution<long>(2, 10)(gen);
    auto rowSize = uniform_int_distribution<long>(20, 80)(gen);
    auto colSize = uniform_int_distribution<long>(20, 80)(gen);

    LOG(INFO) << "running on " << batchSize
              << " x " << numPlanes
              << " x " << rowSize
              << " x " << colSize;

    // Bias should work on non-contiguous formats as well; this may
    // make a non-contiguous tensor
    auto output = makeOutputTensor(batchSize, numPlanes, rowSize, colSize);
    auto bias = makeBiasTensor(numPlanes);
    auto expectedResult = referenceBiasUpdateOutput(output, bias);

    auto outputCuda = copyToCuda(output);
    auto biasCuda = copyToCuda(bias);

    updateOutputBias(outputCuda.get(), biasCuda.get());

    auto result = copyFromCuda(outputCuda.get());

    auto comparison = test::compareTensors(expectedResult, result);
    EXPECT_TRUE(comparison.first) << comparison.second;
  }
}

TEST(BiasTest, testAccGradParameters) {
  random_device dev;
  mt19937 gen(dev());

  for (int run = 0; run < kRuns; ++run) {
    auto batchSize = uniform_int_distribution<long>(1, 200)(gen);
    auto numPlanes = uniform_int_distribution<long>(2, 30)(gen);
    auto rowSize = uniform_int_distribution<long>(10, 100)(gen);
    auto colSize = uniform_int_distribution<long>(10, 100)(gen);
    auto biasScale = uniform_real_distribution<float>(0.1f, 1.0f)(gen);

    LOG(INFO) << "running on " << batchSize
              << " x " << numPlanes
              << " x " << rowSize
              << " x " << colSize;

    // Bias should work on non-contiguous formats as well; this may
    // make a non-contiguous tensor
    auto output = makeOutputTensor(batchSize, numPlanes, rowSize, colSize);
    auto gradBias = makeBiasTensor(numPlanes);

    auto expectedResult =
      referenceBiasAccGradParameters(output, gradBias, biasScale);

    auto outputCuda = copyToCuda(output);
    auto gradBiasCuda = copyToCuda(gradBias);

    accGradParametersBias(outputCuda.get(),
                          gradBiasCuda.get(),
                          biasScale);

    auto result = copyFromCuda(gradBiasCuda.get());

    // Due to order of reductions, our implementation is a little off
    auto comparison = test::compareTensors(expectedResult, result, 5e-4f);
    EXPECT_TRUE(comparison.first) << comparison.second;
  }
}

} } } } // namespace

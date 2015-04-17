// Copyright 2004-present Facebook. All Rights Reserved.

#include "TestUtils.h"
#include "THCTensor.h"

#include "torch/fb/fbcunn/src/DeviceTensorUtils.h"
#include "torch/fb/fbcunn/src/ConvolutionBias.cuh"

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <random>

using namespace std;
using namespace facebook::deeplearning::torch;
using namespace thpp;

namespace facebook { namespace deeplearning { namespace torch { namespace bias {

constexpr int kRuns = 10;

namespace {

// Construct a feature map tensor; bias code should work on
// non-contiguous tensors, so we may create a non-contiguous layout
// for the tensor
Tensor<float>
makeOutputTensor(long batches, long planes, long rows, long cols,
                 unsigned seed) {
  mt19937 gen(seed);

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


// Reference forward bias pass
Tensor<float>
referenceBiasUpdateOutputTemporal(const Tensor<float>& output,
                          const Tensor<float>& bias) {
  auto result = Tensor<float>();
  result.resizeAs(output);
  result.copy(output);

  for (long batch = 0; batch < output.size(0); ++batch) {
    for (long time = 0; time < output.size(2); ++time) {
      float biasVal = bias.at({time});
      for (long plane = 0; plane < output.size(1); ++plane) {
        result.at({batch, plane, time}) += biasVal;
      }
    }
  }
  return result;
}


// Reference accGrad bias pass
Tensor<float>
referenceBiasAccGradParametersTemporal(const Tensor<float>& output,
                               const Tensor<float>& bias,
                               float biasScale) {
  auto result = Tensor<float>();
  result.resizeAs(bias);
  result.copy(bias);

  for (long time = 0; time < output.size(2); ++time) {
    float sum = 0.0f;
    for (long batch = 0; batch < output.size(0); ++batch) {
      for (long plane = 0; plane < output.size(1); ++plane) {
        sum += output.at({batch, plane, time});
      }
    }
    result.at({time}) = sum * biasScale;
  }
  return result;
}

void testOneAccGradParameters(long batchSize,
                              long numPlanes,
                              long rowSize,
                              long colSize,
                              float biasScale,
                              int nRep,
                              unsigned seed) {
  LOG(INFO) << "running on " << batchSize
            << " x " << numPlanes
            << " x " << rowSize
            << " x " << colSize;

  // Bias should work on non-contiguous formats as well; this may
  // make a non-contiguous tensor
  auto output = makeOutputTensor(batchSize, numPlanes, rowSize, colSize, seed);
  auto gradBias = makeBiasTensor(numPlanes);

  auto expectedResult =
    referenceBiasAccGradParameters(output, gradBias, biasScale);

  auto outputCuda = copyToCuda(nullptr, output);
  auto gradBiasCuda = copyToCuda(nullptr, gradBias);

  for (int i = 0; i < nRep; i++) {
    accGradParametersBias(nullptr,
                          outputCuda.get(),
                          gradBiasCuda.get(),
                          biasScale);
  }

  auto result = copyFromCuda(nullptr, gradBiasCuda.get());

  // Due to order of reductions, our implementation is a little off
  auto comparison = test::compareTensors(expectedResult, result, 5e-4f);
  EXPECT_TRUE(comparison.first) << comparison.second;
}

void testOneAccGradParametersTemporal(long batchSize,
                                      long numPlanes,
                                      long timeSize,
                                      float biasScale,
                                      int nRep,
                                      unsigned seed) {
  LOG(INFO) << "running on " << batchSize
            << " x " << numPlanes
            << " x " << timeSize;

  // Bias should work on non-contiguous formats as well; this may
  // make a non-contiguous tensor
  auto output = makeOutputTensor(batchSize, numPlanes, timeSize, 1, seed);
  output.select(3, 0);
  auto gradBias = makeBiasTensor(timeSize);

  auto expectedResult =
    referenceBiasAccGradParametersTemporal(output, gradBias, biasScale);

  auto outputCuda = copyToCuda(nullptr, output);
  auto gradBiasCuda = copyToCuda(nullptr, gradBias);

  for (int i = 0; i < nRep; i++) {
    accGradParametersTemporalBias(nullptr,
                                  outputCuda.get(),
                                  gradBiasCuda.get(),
                                  biasScale);
  }

  auto result = copyFromCuda(nullptr, gradBiasCuda.get());

  auto comparison = test::compareTensors(expectedResult, result, 5e-4f);
  EXPECT_TRUE(comparison.first) << comparison.second;
}

} // namespace



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
    auto output = makeOutputTensor(
      batchSize, numPlanes, rowSize, colSize, dev());
    auto bias = makeBiasTensor(numPlanes);
    auto expectedResult = referenceBiasUpdateOutput(output, bias);

    auto outputCuda = copyToCuda(nullptr, output);
    auto biasCuda = copyToCuda(nullptr, bias);

    updateOutputBias(nullptr, outputCuda.get(), biasCuda.get());

    auto result = copyFromCuda(nullptr, outputCuda.get());

    auto comparison = test::compareTensors(expectedResult, result);
    EXPECT_TRUE(comparison.first) << comparison.second;
  }
}

TEST(BiasTest, testUpdateOutputTemporal) {
  random_device dev;
  mt19937 gen(dev());

  for (int run = 0; run < kRuns; ++run) {
    auto batchSize = uniform_int_distribution<long>(5, 15)(gen);
    auto numPlanes = uniform_int_distribution<long>(2, 10)(gen);
    auto timeSize = uniform_int_distribution<long>(20, 80)(gen);

    LOG(INFO) << "running on " << batchSize
              << " x " << numPlanes
              << " x " << timeSize;

    // Bias should work on non-contiguous formats as well; this may
    // make a non-contiguous tensor
    auto output = makeOutputTensor(batchSize, numPlanes, timeSize, 1, dev());
    output.select(3, 0);
    auto bias = makeBiasTensor(timeSize);
    Tensor<float> transposedOutput;
    auto expectedResult = referenceBiasUpdateOutputTemporal(output, bias);

    auto outputCuda = copyToCuda(nullptr, output);
    auto biasCuda = copyToCuda(nullptr, bias);

    updateOutputTemporalBias(nullptr, outputCuda.get(), biasCuda.get());

    auto result = copyFromCuda(nullptr, outputCuda.get());

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

    testOneAccGradParameters(
      batchSize, numPlanes, rowSize, colSize, biasScale, 1, dev());
  }
}

TEST(BiasTest, testAccGradParametersTemporal) {
  random_device dev;
  mt19937 gen(dev());

  for (int run = 0; run < kRuns; ++run) {
    auto batchSize = uniform_int_distribution<long>(1, 200)(gen);
    auto numPlanes = uniform_int_distribution<long>(2, 30)(gen);
    auto timeSize = uniform_int_distribution<long>(10, 100)(gen);
    auto biasScale = uniform_real_distribution<float>(0.1f, 1.0f)(gen);

    testOneAccGradParametersTemporal(
      batchSize, numPlanes, timeSize, biasScale, 1, dev());
  }
}

// better to use folly::benchmark but it's overkill for right now
//
// nvprof times in kernel:
// batchTiles = min(output.getSize(0), 32)       -> 40us
// batchTiles = 1                                -> 175us
// (unrolled 8x)                                 -> 190us
TEST(BiasTest, benchmarkAccGradParameters) {
  constexpr int kRep = 1000;
  testOneAccGradParameters(64, 16, 31, 31, 0.5f, kRep, 0);
}

} } } } // namespace

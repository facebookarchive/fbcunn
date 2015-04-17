// Copyright 2004-present Facebook. All Rights Reserved.

#include "torch/fb/fbcunn/src/DeviceTensorUtils.h"
#include "THCTensor.h"
#include "torch/fb/fbcunn/src/fft/Utils.h"
#include "torch/fb/fbcunn/src/fft/CuFFTConvolution_UpdateOutput.cuh"
#include "torch/fb/fbcunn/src/fft/CuFFTConvolution_AccGradParameters.cuh"
#include "torch/fb/fbcunn/src/fft/CuFFTConvolution_UpdateGradInput.cuh"
#include "torch/fb/fbcunn/test/InputCentricConvolution_UpdateOutput.cuh"
#include "torch/fb/fbcunn/test/ReferenceConvolutions.h"
#include "torch/fb/fbcunn/test/TestUtils.h"

#include <cuda.h>
#include <folly/Optional.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <tuple>

using namespace std;
using namespace facebook::deeplearning::torch;

DEFINE_bool(verify, true, "Run the convolution and verify the output");
DEFINE_bool(debug, false, "Print basic information on tensors");

// Override gtest_main so as to parse the --verify flag
int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  google::ParseCommandLineFlags(&argc, &argv, true);
  return RUN_ALL_TESTS();
}

namespace facebook { namespace deeplearning { namespace torch { namespace test {

// Default accGradParameters bias for testing
constexpr float kAccGradBias = 0.333f;

struct Params {
  constexpr Params(long batchVal,
                   long inputPlanesVal,
                   long inputRowVal,
                   long inputColVal,
                   long outputPlanesVal,
                   long filterSizeVal,
                   long filterStrideVal)
      : batch(batchVal),
        inputPlanes(inputPlanesVal),
        inputRow(inputRowVal),
        inputCol(inputColVal),
        outputPlanes(outputPlanesVal),
        filterSize(filterSizeVal),
        filterStride(filterStrideVal) {
  }

  long outputRow() const {
    return getValidConvSize(inputRow, filterSize, filterStride);
  }

  long outputCol() const {
    return getValidConvSize(inputCol, filterSize, filterStride);
  }

  const long batch;
  const long inputPlanes;
  const long inputRow;
  const long inputCol;
  const long outputPlanes;
  const long filterSize;
  const long filterStride;
};

namespace {

constexpr float kBatchFactor = 0.1f;
constexpr float kFilterFactor = 0.2f;
constexpr float kPlaneFactor = 0.3f;
constexpr float kInputRowFactor = 0.4f;
constexpr float kInputColFactor = 0.5f;
constexpr float kOutputRowFactor = 0.6f;
constexpr float kOutputColFactor = 0.7f;
constexpr float kFilterRowFactor = 0.8f;
constexpr float kFilterColFactor = 0.9f;

// Construct a verification input tensor
// Input tensor dimensions:
// 0: num images in batch
// 1: num image planes
// 2: num image rows
// 3: num image cols
Tensor<float>
makeInputTensor(
  long batches, long planes, long rows, long cols,
  const folly::Optional<tuple<long, long, long, long>>& padding = folly::none) {
  return makeTestTensor(
    {batches, planes, rows, cols},
    {kBatchFactor, kPlaneFactor, kInputRowFactor, kInputColFactor},
    padding);
}

// Construct a verification output tensor
// Output tensor dimensions:
// 0: num images in batch
// 1: num filter planes
// 2: num output rows
// 3: num output cols
Tensor<float>
makeOutputTensor(
  long batches, long planes, long rows, long cols,
  const folly::Optional<tuple<long, long, long, long>>& padding = folly::none) {
  return makeTestTensor(
    {batches, planes, rows, cols},
    {kBatchFactor, kPlaneFactor, kOutputRowFactor, kOutputColFactor},
    padding);
}

// Construct a verification filter tensor
// Filter tensor dimensions:
// 0: num filters
// 1: num image planes
// 2: num filter rows
// 3: num filter cols
Tensor<float>
makeFilterTensor(
  long filters, long planes, long rows, long cols,
  const folly::Optional<tuple<long, long, long, long>>& padding = folly::none) {
  return makeTestTensor(
    {filters, planes, rows, cols},
    {kFilterFactor, kPlaneFactor, kFilterRowFactor, kFilterColFactor},
    padding);
}

// Class to test different forms of convolution
class ConvolutionModule {
 public:
  ConvolutionModule() {
  }

  virtual ~ConvolutionModule() {
  }

  // Test a single combination of (input, output) -> filters
  void testUpdateOutput(
    Tensor<float>& input,
    Tensor<float>& filters,
    long filterRowStride,
    long filterColStride,
    const folly::Optional<tuple<long, long, long, long>>& inputPadding) {
    if (FLAGS_verify) {
      auto expectedOutput = crossCorrelationValidOnly(input,
                                                      filters,
                                                      filterRowStride,
                                                      filterColStride,
                                                      inputPadding);

      auto output = Tensor<float>{{expectedOutput.size(0),
                                   expectedOutput.size(1),
                                   expectedOutput.size(2),
                                   expectedOutput.size(3)}};
      output.fill(0);

      updateOutput(
        input, filters, filterRowStride, filterColStride, inputPadding, output);

      auto comparison = compareTensors(expectedOutput, output);
      EXPECT_TRUE(comparison.first) << comparison.second;
    } else {
      auto output = Tensor<float>{
        {input.size(0),
         filters.size(0),
         getValidConvSize(
           input.size(2) +
           (inputPadding ? get<0>(*inputPadding) + get<1>(*inputPadding) : 0),
           filters.size(2),
           filterRowStride),
         getValidConvSize(
           input.size(3) +
           (inputPadding ? get<2>(*inputPadding) + get<3>(*inputPadding) : 0),
           filters.size(3),
           filterColStride)}};
      output.fill(0);

      updateOutput(
        input, filters, filterRowStride, filterColStride, inputPadding, output);
    }

    // In case the implementation used CUDA, verify that kernel launch
    // did not fail
    EXPECT_EQ(cudaSuccess, cudaDeviceSynchronize());
    EXPECT_EQ(cudaSuccess, cudaGetLastError());
  }

  // Test a single combination of (output, filters) -> input
  void testUpdateGradInput(
    Tensor<float>& output,
    Tensor<float>& filters,
    long filterRowStride,
    long filterColStride,
    const folly::Optional<tuple<long, long, long, long>>& inputPadding) {
    if (FLAGS_verify) {
      auto expectedInput = convolutionFull(output,
                                           filters,
                                           filterRowStride,
                                           filterColStride,
                                           inputPadding);

      auto input = Tensor<float>{{expectedInput.size(0),
                                  expectedInput.size(1),
                                  expectedInput.size(2),
                                  expectedInput.size(3)}};
      input.fill(0);

      updateGradInput(
        output, filters, filterRowStride, filterColStride, inputPadding, input);

      auto comparison = compareTensors(expectedInput, input);
      EXPECT_TRUE(comparison.first) << comparison.second;
    } else {
      auto input = Tensor<float>{
        {output.size(0),
         filters.size(1),
         getFullConvSize(output.size(2), filters.size(2), filterRowStride) -
         (inputPadding ? get<0>(*inputPadding) + get<1>(*inputPadding) : 0),
         getFullConvSize(output.size(3), filters.size(3), filterColStride) -
         (inputPadding ? get<2>(*inputPadding) + get<3>(*inputPadding) : 0)}};
      input.fill(0);

      updateGradInput(
        output, filters, filterRowStride, filterColStride, inputPadding, input);
    }

    // In case the implementation used CUDA, verify that kernel launch
    // did not fail
    EXPECT_EQ(cudaSuccess, cudaDeviceSynchronize());
    EXPECT_EQ(cudaSuccess, cudaGetLastError());
  }

  // Test a single combination of (input, output) -> filters
  void testAccGradParameters(
    Tensor<float>& input,
    Tensor<float>& output,
    long filterRowStride,
    long filterColStride,
    float scale,
    const folly::Optional<tuple<long, long, long, long>>& inputPadding) {
    if (FLAGS_verify) {
      auto expectedFilters = crossCorrelationReverseValidOnly(input,
                                                              output,
                                                              filterRowStride,
                                                              filterColStride,
                                                              scale,
                                                              inputPadding);

      auto filters = Tensor<float>{{expectedFilters.size(0),
                                    expectedFilters.size(1),
                                    expectedFilters.size(2),
                                    expectedFilters.size(3)}};
      filters.fill(0);

      accGradParameters(input, output,
                        filterRowStride, filterColStride,
                        scale, inputPadding, filters);

      auto comparison = compareTensors(expectedFilters, filters);
      EXPECT_TRUE(comparison.first) << comparison.second;
    } else {
      auto filters = Tensor<float>{
        {output.size(1),
         input.size(1),
         getValidRevConvSize(
           input.size(2) +
           (inputPadding ? get<0>(*inputPadding) + get<1>(*inputPadding) : 0),
           output.size(2),
           filterRowStride),
         getValidRevConvSize(
           input.size(3) +
           (inputPadding ? get<2>(*inputPadding) + get<3>(*inputPadding) : 0),
           output.size(3),
           filterColStride)}};
      filters.fill(0);

      accGradParameters(input, output,
                        filterRowStride, filterColStride,
                        scale, inputPadding, filters);
    }

    // In case the implementation used CUDA, verify that kernel launch
    // did not fail
    EXPECT_EQ(cudaSuccess, cudaDeviceSynchronize());
    EXPECT_EQ(cudaSuccess, cudaGetLastError());
  }

  virtual void updateOutput(
    Tensor<float>& input,
    Tensor<float>& filters,
    long filterRowStride,
    long filterColStride,
    const folly::Optional<tuple<long, long, long, long>>& inputPadding,
    Tensor<float>& output) = 0;

  virtual void updateGradInput(
    Tensor<float>& output,
    Tensor<float>& filters,
    long filterRowStride,
    long filterColStride,
    const folly::Optional<tuple<long, long, long, long>>& inputPadding,
    Tensor<float>& input) = 0;

  virtual void accGradParameters(
    Tensor<float>& input,
    Tensor<float>& output,
    long filterRowStride,
    long filterColStride,
    float scale,
    const folly::Optional<tuple<long, long, long, long>>& inputPadding,
    Tensor<float>& filters) = 0;
};

// Test to make sure that asymmetric padding in the reference
// implementation works, when compared to the non-padded reference
// version
class ReferenceAsymmetricPaddingTest : public ConvolutionModule {
 public:
  ReferenceAsymmetricPaddingTest(long topPadding,
                                 long bottomPadding,
                                 long leftPadding,
                                 long rightPadding)
      : testPadding_(topPadding, bottomPadding, leftPadding, rightPadding) {
  }

  void updateOutput(
    Tensor<float>& input,
    Tensor<float>& filters,
    long filterRowStride,
    long filterColStride,
    const folly::Optional<tuple<long, long, long, long>>& inputPadding,
    Tensor<float>& output) override {
    ASSERT_FALSE(inputPadding); // This is the padding that is
                                // provided to the reference

    const auto topPad = get<0>(testPadding_);
    const auto bottomPad = get<1>(testPadding_);
    const auto leftPad = get<2>(testPadding_);
    const auto rightPad = get<3>(testPadding_);

    // The row padding must be a multiple of the row stride
    ASSERT_EQ(0, topPad % filterRowStride);
    ASSERT_EQ(0, bottomPad % filterRowStride);

    // The col padding must be a multiple of the col stride
    ASSERT_EQ(0, leftPad % filterColStride);
    ASSERT_EQ(0, rightPad % filterColStride);

    // Use the padding that the test is configured for, and then we'll
    // narrow the output below
    auto asymPadOutput =
      crossCorrelationValidOnly(input,
                                filters,
                                filterRowStride,
                                filterColStride,
                                testPadding_);

    // In order for the output to be comparable via narrowing the
    // tensor, the paddings have to be multiples of the stride as
    // verified above, so this is legitimate
    const auto topPadInOutput = topPad / filterRowStride;
    const auto bottomPadInOutput = bottomPad / filterRowStride;
    const auto leftPadInOutput = leftPad / filterColStride;
    const auto rightPadInOutput = rightPad / filterColStride;

    asymPadOutput.narrow(
      2, topPadInOutput,
      asymPadOutput.size(2) - topPadInOutput - bottomPadInOutput);
    asymPadOutput.narrow(
      3, leftPadInOutput,
      asymPadOutput.size(3) - leftPadInOutput - rightPadInOutput);

    output = std::move(asymPadOutput);
  }

  void updateGradInput(
    Tensor<float>& output,
    Tensor<float>& filters,
    long filterRowStride,
    long filterColStride,
    const folly::Optional<tuple<long, long, long, long>>& inputPadding,
    Tensor<float>& input) override {
    ASSERT_FALSE(inputPadding); // This is the padding that is
                                // provided to the reference
    // The row padding must be a multiple of the row stride
    ASSERT_EQ(0, get<0>(testPadding_) % filterRowStride);
    ASSERT_EQ(0, get<1>(testPadding_) % filterRowStride);

    // The col padding must be a multiple of the col stride
    ASSERT_EQ(0, get<2>(testPadding_) % filterColStride);
    ASSERT_EQ(0, get<3>(testPadding_) % filterColStride);

    auto outputPadding =
      make_tuple(get<0>(testPadding_) / filterRowStride,
                 get<1>(testPadding_) / filterRowStride,
                 get<2>(testPadding_) / filterColStride,
                 get<3>(testPadding_) / filterColStride);

    // Construct a padded version of the same reference output for
    // comparison, since we can't evaluate this through just narrow()
    auto outputPadded =
      makeOutputTensor(output.size(0), filters.size(0),
                       output.size(2), output.size(3), outputPadding);

    input = convolutionFull(outputPadded,
                            filters,
                            filterRowStride,
                            filterColStride,
                            testPadding_);
  }

  void accGradParameters(
    Tensor<float>& input,
    Tensor<float>& output,
    long filterRowStride,
    long filterColStride,
    float scale,
    const folly::Optional<tuple<long, long, long, long>>& inputPadding,
    Tensor<float>& filters) override {
    ASSERT_FALSE(inputPadding); // This is the padding that is
                                // provided to the reference
    // The row padding must be a multiple of the row stride
    ASSERT_EQ(0, get<0>(testPadding_) % filterRowStride);
    ASSERT_EQ(0, get<1>(testPadding_) % filterRowStride);

    // The col padding must be a multiple of the col stride
    ASSERT_EQ(0, get<2>(testPadding_) % filterColStride);
    ASSERT_EQ(0, get<3>(testPadding_) % filterColStride);

    auto outputPadding =
      make_tuple(get<0>(testPadding_) / filterRowStride,
                 get<1>(testPadding_) / filterRowStride,
                 get<2>(testPadding_) / filterColStride,
                 get<3>(testPadding_) / filterColStride);

    // Construct a padded version of the same reference output for comparison
    auto outputPadded =
      makeOutputTensor(input.size(0), output.size(1),
                       output.size(2), output.size(3), outputPadding);

    filters = crossCorrelationReverseValidOnly(input,
                                               outputPadded,
                                               filterRowStride,
                                               filterColStride,
                                               scale,
                                               testPadding_);
  }

 private:
  const tuple<long, long, long, long> testPadding_;
};

class TorchTest : public ConvolutionModule {
 public:
  void updateOutput(
    Tensor<float>& input,
    Tensor<float>& filters,
    long filterRowStride,
    long filterColStride,
    const folly::Optional<tuple<long, long, long, long>>& inputPadding,
    Tensor<float>& output) override {
    ASSERT_FALSE(inputPadding); // padding not supported

    // Torch APIs don't take input/filters as const even though they
    // effectively are
    auto inputTH = input.moveAsTH();
    auto filtersTH = filters.moveAsTH();
    auto outputTH = output.moveAsTH();

    THFloatTensor_conv2Dmm(outputTH, 1.0, 1.0, inputTH, filtersTH,
                           filterRowStride, filterColStride, "V", "X");

    // Rebind for evaluation and cleanup
    output = std::move(outputTH);
    input = std::move(inputTH);
    filters = std::move(filtersTH);
  }

  void updateGradInput(
    Tensor<float>& output,
    Tensor<float>& filters,
    long filterRowStride,
    long filterColStride,
    const folly::Optional<tuple<long, long, long, long>>& inputPadding,
    Tensor<float>& input) override {
    ASSERT_FALSE(inputPadding); // padding not supported

    auto inputTH = input.moveAsTH();
    auto outputTH = output.moveAsTH();

    // Torch requires transposition of filters
    filters.transpose(0, 1);
    auto filtersTH = filters.moveAsTH();

    THFloatTensor_conv2Dmm(inputTH, 0.0, 1.0, outputTH, filtersTH,
                           filterRowStride, filterColStride, "F", "C");

    // Rebind for evaluation and cleanup
    input = std::move(inputTH);
    output = std::move(outputTH);
    filters = std::move(filtersTH);
  }

  void accGradParameters(
    Tensor<float>& input,
    Tensor<float>& output,
    long filterRowStride,
    long filterColStride,
    float scale,
    const folly::Optional<tuple<long, long, long, long>>& inputPadding,
    Tensor<float>& filters) override {
    ASSERT_FALSE(inputPadding); // padding not supported

    auto inputTH = input.moveAsTH();
    auto outputTH = output.moveAsTH();
    auto filtersTH = filters.moveAsTH();

    THFloatTensor_conv2DRevgerm(filtersTH, 1.0, scale,
                                inputTH, outputTH,
                                filterRowStride, filterColStride);

    // Rebind for evaluation and cleanup
    input = std::move(inputTH);
    output = std::move(outputTH);
    filters = std::move(filtersTH);
  }
};

class ReferenceInputCentricTest : public ConvolutionModule {
 public:
  void updateOutput(
    Tensor<float>& input,
    Tensor<float>& filters,
    long filterRowStride,
    long filterColStride,
    const folly::Optional<tuple<long, long, long, long>>& inputPadding,
    Tensor<float>& output) override {

    output =
      crossCorrelationValidOnlyInputCentric(input,
                                            filters,
                                            filterRowStride,
                                            filterColStride,
                                            inputPadding);
  }

  void updateGradInput(
    Tensor<float>& output,
    Tensor<float>& filters,
    long filterRowStride,
    long filterColStride,
    const folly::Optional<tuple<long, long, long, long>>& inputPadding,
    Tensor<float>& input) override {
    ASSERT_TRUE(false) << "not implemented";
  }

  void accGradParameters(
    Tensor<float>& input,
    Tensor<float>& output,
    long filterRowStride,
    long filterColStride,
    float scale,
    const folly::Optional<tuple<long, long, long, long>>& inputPadding,
    Tensor<float>& filters) override {
    ASSERT_TRUE(false) << "not implemented";
  }
};


class InputCentricTest : public ConvolutionModule {
 public:
  enum class Layout : bool {
    Normal = false,
    Relayout = true
  };

  explicit InputCentricTest(Layout l = Layout::Relayout) : layout(l) { }

  void updateOutput(
    Tensor<float>& input,
    Tensor<float>& filters,
    long filterRowStride,
    long filterColStride,
    const folly::Optional<tuple<long, long, long, long>>& inputPadding,
    Tensor<float>& output) override {
    ASSERT_FALSE(inputPadding); // padding not supported

    auto inputCuda = copyToCuda(nullptr, input);
    auto filtersCuda = copyToCuda(nullptr, filters);

    CHECK(layout == Layout::Relayout) <<
      "Only Relayout mode is supported for this kernel atm";

    // Relayout filters, for instance for 96 x 3 x 11 x 11 we get
    auto filtersTmp = Tensor<float> ({
        filters.size(1),     // 3
          filters.size(2),   // 11
          filters.size(3),   // 11
          filters.size(0)}); // 96
    for (long i = 0; i < filters.size(0); ++i) {
      for (long j = 0; j < filters.size(1); ++j) {
        for (long k = 0; k < filters.size(2); ++k) {
          for (long l = 0; l < filters.size(3); ++l) {
            filtersTmp.at({j, k, l, i}) = filters.at({i, j, k, l});
          }
        }
      }
    }
    auto filtersCudaTmp = copyToCuda(nullptr, filtersTmp);

    // Relayout output, for instance for 32 x 96 x 71 x 71 we get
    const int filterRowSize = filters.size(2);
    const int ceilFilterSizeFilterStride =
      (filterRowSize + filterRowStride - 1) / filterRowStride;
    auto outputCudaTmp = makeTHCudaTensorFull(nullptr, {
        output.size(0),      // 32
          // 71 + 2 * ceilFilterSizeFilterStride
          // This expansion by 2 * ceilFilterSizeFilterStride allows us to
          // overwrite in the kernel without hitting out of bounds and gives
          // another 10% speedup
          output.size(2) + 2 * ceilFilterSizeFilterStride,
          output.size(3),    // 71
          output.size(1)}    // 96
      );

    bool result =
      InputCentricRelayoutConvolution_UpdateOutput(nullptr,
                                                   inputCuda.get(),
                                                   filtersCudaTmp.get(),
                                                   filterRowStride,
                                                   filterColStride,
                                                   outputCudaTmp.get());

    EXPECT_TRUE(result);

    // Recover actual output from layout
    auto outputTmp = copyFromCuda(nullptr, outputCudaTmp.get());
    for (long i = 0; i < output.size(0); ++i) {
      for (long j = 0; j < output.size(1); ++j) {
        for (long k = 0; k < output.size(2); ++k) {
          for (long l = 0; l < output.size(3); ++l) {
            output.at({i, j, k, l}) =
              outputTmp.at({i, k + ceilFilterSizeFilterStride, l, j});
          }
        }
      }
    }
  }

  void updateGradInput(
    Tensor<float>& output,
    Tensor<float>& filters,
    long filterRowStride,
    long filterColStride,
    const folly::Optional<tuple<long, long, long, long>>& inputPadding,
    Tensor<float>& input) override {
    ASSERT_TRUE(false) << "not implemented";
  }

  void accGradParameters(
    Tensor<float>& input,
    Tensor<float>& output,
    long filterRowStride,
    long filterColStride,
    float scale,
    const folly::Optional<tuple<long, long, long, long>>& inputPadding,
    Tensor<float>& filters) override {
    ASSERT_TRUE(false) << "not implemented";
  }

 private:
  Layout layout;
};


class CuFFT : public ConvolutionModule {
 public:
  static const int kFFTDims = 2;
  enum class Implementation : bool { Reference, Optimized };

  explicit CuFFT(
    Implementation impl = Implementation::Optimized) :
      impl_(impl) {}

  void updateOutput(
    Tensor<float>& input,
    Tensor<float>& filters,
    long filterRowStride,
    long filterColStride,
    const folly::Optional<tuple<long, long, long, long>>& inputPadding,
    Tensor<float>& output) override {
    ASSERT_FALSE(inputPadding); // padding not supported

    if (FLAGS_debug) {
      LOG(INFO) << "input: " << input;
      LOG(INFO) << "filters: " << filters;
      LOG(INFO) << "output: " << output;
    }

    // Zero-Padding
    int maxRows = std::max(input.size(2),
      std::max(filters.size(2), output.size(2)));
    int maxCols = std::max(input.size(3),
      std::max(filters.size(3), output.size(3)));

    std::vector<long> maxSizes({maxRows, maxCols});
    auto realComplexPair = makeCuFFTTensors<kFFTDims>(nullptr, input, maxSizes);
    auto inputTHCudaTensor = std::move(realComplexPair.first);
    auto inputComplexTHCudaTensor = std::move(realComplexPair.second);
    auto inputComplexTHCudaTensorT = makeCuFFTTensorComplex<kFFTDims>(
      nullptr, inputTHCudaTensor.get(), maxSizes);

    realComplexPair =
      makeCuFFTTensors<kFFTDims>(nullptr, filters, maxSizes);
    auto filtersTHCudaTensor = std::move(realComplexPair.first);
    auto filtersComplexTHCudaTensor = std::move(realComplexPair.second);
    auto filtersComplexTHCudaTensorT = makeCuFFTTensorComplex<kFFTDims>(
      nullptr, filtersTHCudaTensor.get(), maxSizes);

    realComplexPair =
      makeCuFFTTensors<kFFTDims>(nullptr, output, maxSizes);
    auto outputTHCudaTensor = std::move(realComplexPair.first);
    auto outputComplexTHCudaTensor = std::move(realComplexPair.second);
    auto outputComplexTHCudaTensorT = makeCuFFTTensorComplex<kFFTDims>(
      nullptr, outputTHCudaTensor.get(), maxSizes);

    // We don't test the bias here
    auto bias = Tensor<float>{{output.size(0)}};
    bias.fill(0);
    auto biasCuda = copyToCuda(nullptr, bias);

    if (impl_ == Implementation::Reference) {
      CuFFTConvolution_ReferenceUpdateOutput(nullptr,
                                             inputTHCudaTensor.get(),
                                             filtersTHCudaTensor.get(),
                                             outputTHCudaTensor.get(),
                                             biasCuda.get(),
                                             inputComplexTHCudaTensor.get(),
                                             filtersComplexTHCudaTensor.get(),
                                             outputComplexTHCudaTensor.get());
    } else {
      CuFFTConvolution_UpdateOutput(nullptr,
                                    inputTHCudaTensor.get(),
                                    filtersTHCudaTensor.get(),
                                    outputTHCudaTensor.get(),
                                    biasCuda.get(),
                                    inputComplexTHCudaTensor.get(),
                                    filtersComplexTHCudaTensor.get(),
                                    outputComplexTHCudaTensor.get(),
                                    inputComplexTHCudaTensorT.get(),
                                    filtersComplexTHCudaTensorT.get(),
                                    outputComplexTHCudaTensorT.get());
    }

    if (FLAGS_verify) {
      checkExpectedInput(input,
                         copyFromCuda(nullptr, inputTHCudaTensor.get()));
      checkExpectedInput(filters,
                         copyFromCuda(nullptr, filtersTHCudaTensor.get()));

      // Recover actual output from padded layout, output is smaller
      // than outputTmp when kernelSize > 1
      auto outputTmp = copyFromCuda(nullptr, outputTHCudaTensor.get());
      for (long i = 0; i < output.size(0); ++i) {
        for (long j = 0; j < output.size(1); ++j) {
          for (long k = 0; k < output.size(2); ++k) {
            for (long l = 0; l < output.size(3); ++l) {
              output.at({i, j, k, l}) = outputTmp.at({i, j, k, l});
            }
          }
        }
      }
    }

    saveOutputTHCudaTensor = std::move(outputTHCudaTensor);
  }

  void accGradParameters(
    Tensor<float>& input,
    Tensor<float>& output,
    long filterRowStride,
    long filterColStride,
    float scale,
    const folly::Optional<tuple<long, long, long, long>>& inputPadding,
    Tensor<float>& filters) override {
    ASSERT_FALSE(inputPadding); // padding not supported

    if (FLAGS_debug) {
      LOG(INFO) << "input: " << input;
      LOG(INFO) << "filters: " << filters;
      LOG(INFO) << "output: " << output;
    }

    // Zero-Padding
    int maxRows = std::max(input.size(2),
      std::max(filters.size(2), output.size(2)));
    int maxCols = std::max(input.size(3),
      std::max(filters.size(3), output.size(3)));

    std::vector<long> maxSizes({maxRows, maxCols});
    auto realComplexPair =
      makeCuFFTTensors<kFFTDims>(nullptr, input, maxSizes);
    auto inputTHCudaTensor = std::move(realComplexPair.first);
    auto inputComplexTHCudaTensor = std::move(realComplexPair.second);
    auto inputComplexTHCudaTensorT = makeCuFFTTensorComplex<kFFTDims>(
      nullptr, inputTHCudaTensor.get(), maxSizes);

    realComplexPair =
      makeCuFFTTensors<kFFTDims>(nullptr, filters, maxSizes);
    auto filtersTHCudaTensor = std::move(realComplexPair.first);
    auto filtersComplexTHCudaTensor = std::move(realComplexPair.second);
    auto filtersComplexTHCudaTensorT = makeCuFFTTensorComplex<kFFTDims>(
      nullptr, filtersTHCudaTensor.get(), maxSizes);

    realComplexPair =
      makeCuFFTTensors<kFFTDims>(nullptr, output, maxSizes);
    auto outputTHCudaTensor = std::move(realComplexPair.first);
    auto outputComplexTHCudaTensor = std::move(realComplexPair.second);
    auto outputComplexTHCudaTensorT = makeCuFFTTensorComplex<kFFTDims>(
      nullptr, outputTHCudaTensor.get(), maxSizes);

    // We don't test the bias here
    auto bias = Tensor<float>{{filters.size(0)}};
    bias.fill(0);
    auto biasCuda = copyToCuda(nullptr, bias);

    if (impl_ == Implementation::Reference) {
      CuFFTConvolution_ReferenceAccGradParameters(
        nullptr,
        inputTHCudaTensor.get(),
        filtersTHCudaTensor.get(),
        outputTHCudaTensor.get(),
        biasCuda.get(),
        scale,
        inputComplexTHCudaTensor.get(),
        filtersComplexTHCudaTensor.get(),
        outputComplexTHCudaTensor.get());
    } else {
      CuFFTConvolution_AccGradParameters(nullptr,
                                         inputTHCudaTensor.get(),
                                         filtersTHCudaTensor.get(),
                                         outputTHCudaTensor.get(),
                                         biasCuda.get(),
                                         scale,
                                         inputComplexTHCudaTensor.get(),
                                         filtersComplexTHCudaTensor.get(),
                                         outputComplexTHCudaTensor.get(),
                                         inputComplexTHCudaTensorT.get(),
                                         filtersComplexTHCudaTensorT.get(),
                                         outputComplexTHCudaTensorT.get());
    }

    if (FLAGS_verify) {
      checkExpectedInput(input,
                         copyFromCuda(nullptr, inputTHCudaTensor.get()));
      checkExpectedInput(output,
                         copyFromCuda(nullptr, outputTHCudaTensor.get()));
      // Recover actual filters from padded layout, filters is smaller
      // than filtersTmp when kernelSize > 1
      auto filtersTmp = copyFromCuda(nullptr, filtersTHCudaTensor.get());
      for (long i = 0; i < filters.size(0); ++i) {
        for (long j = 0; j < filters.size(1); ++j) {
          for (long k = 0; k < filters.size(2); ++k) {
            for (long l = 0; l < filters.size(3); ++l) {
              filters.at({i, j, k, l}) = filtersTmp.at({i, j, k, l});
            }
          }
        }
      }
    }
  }

  void updateGradInput(
    Tensor<float>& output,
    Tensor<float>& filters,
    long filterRowStride,
    long filterColStride,
    const folly::Optional<tuple<long, long, long, long>>& inputPadding,
    Tensor<float>& input) override {
    ASSERT_FALSE(inputPadding); // padding not supported

    if (FLAGS_debug) {
      LOG(INFO) << "input: " << input;
      LOG(INFO) << "filters: " << filters;
      LOG(INFO) << "output: " << output;
    }

    // Zero-Padding
    int maxRows = std::max(input.size(2),
      std::max(filters.size(2), output.size(2)));
    int maxCols = std::max(input.size(3),
      std::max(filters.size(3), output.size(3)));

    std::vector<long> maxSizes({maxRows, maxCols});
    auto realComplexPair =
      makeCuFFTTensors<kFFTDims>(nullptr, input, maxSizes);
    auto inputTHCudaTensor = std::move(realComplexPair.first);
    auto inputComplexTHCudaTensor = std::move(realComplexPair.second);
    auto inputComplexTHCudaTensorT = makeCuFFTTensorComplex<kFFTDims>(
      nullptr, inputTHCudaTensor.get(), maxSizes);

    realComplexPair =
      makeCuFFTTensors<kFFTDims>(nullptr, filters, maxSizes);
    auto filtersTHCudaTensor = std::move(realComplexPair.first);
    auto filtersComplexTHCudaTensor = std::move(realComplexPair.second);
    auto filtersComplexTHCudaTensorT =
      makeCuFFTTensorComplex<kFFTDims>(
        nullptr, filtersTHCudaTensor.get(), maxSizes);

    realComplexPair =
      makeCuFFTTensors<kFFTDims>(nullptr, output, maxSizes);
    auto outputTHCudaTensor = std::move(realComplexPair.first);
    auto outputComplexTHCudaTensor = std::move(realComplexPair.second);
    auto outputComplexTHCudaTensorT = makeCuFFTTensorComplex<kFFTDims>(
      nullptr, outputTHCudaTensor.get(), maxSizes);

    if (impl_ == Implementation::Reference) {
      CuFFTConvolution_ReferenceUpdateGradInput(
        nullptr,
        inputTHCudaTensor.get(),
        filtersTHCudaTensor.get(),
        outputTHCudaTensor.get(),
        inputComplexTHCudaTensor.get(),
        filtersComplexTHCudaTensor.get(),
        outputComplexTHCudaTensor.get());
    } else {
      CuFFTConvolution_UpdateGradInput(
        nullptr,
        inputTHCudaTensor.get(),
        filtersTHCudaTensor.get(),
        outputTHCudaTensor.get(),
        inputComplexTHCudaTensor.get(),
        filtersComplexTHCudaTensor.get(),
        outputComplexTHCudaTensor.get(),
        inputComplexTHCudaTensorT.get(),
        filtersComplexTHCudaTensorT.get(),
        outputComplexTHCudaTensorT.get());
    }

    if (FLAGS_verify) {
      checkExpectedInput(filters,
                         copyFromCuda(nullptr, filtersTHCudaTensor.get()));
      checkExpectedInput(output,
                         copyFromCuda(nullptr, outputTHCudaTensor.get()));
      // Recover actual filters from padded layout, filters is smaller
      // than filtersTmp when kernelSize > 1
      auto inputTmp = copyFromCuda(nullptr, inputTHCudaTensor.get());
      for (long i = 0; i < input.size(0); ++i) {
        for (long j = 0; j < input.size(1); ++j) {
          for (long k = 0; k < input.size(2); ++k) {
            for (long l = 0; l < input.size(3); ++l) {
              input.at({i, j, k, l}) = inputTmp.at({i, j, k, l});
            }
          }
        }
      }
    }

    saveInputTHCudaTensor = std::move(inputTHCudaTensor);
  }

  std::unique_ptr<THCudaTensor, CudaTensorDeleter> saveOutputTHCudaTensor;
  std::unique_ptr<THCudaTensor, CudaTensorDeleter> saveInputTHCudaTensor;

  static void checkExpectedInput(Tensor<float> expectedInput,
                                 Tensor<float> actualInput,
                                 float relativeError = 5e-4f) {
    auto comparison =
      compareTensors(expectedInput, actualInput, relativeError, 10, true);
    ASSERT_TRUE(comparison.first) << "Mismatch " << comparison.second;
  }

 private:
  Implementation impl_;
};

} // unnamed namespace

//
// Reference asymmetric padding tests (comparing the reference
// implementation with asymmetric padding against the one with no
// padding)
//

constexpr Params kRAPParams(2, 2, 6, 6, 2, 4, 2);

TEST(CudaConvolutionTest, ReferenceAsymmetricPadding_updateOutput) {
  auto input =
    makeInputTensor(kRAPParams.batch, kRAPParams.inputPlanes,
                    kRAPParams.inputRow, kRAPParams.inputCol);
  auto filters =
    makeFilterTensor(kRAPParams.outputPlanes, kRAPParams.inputPlanes,
                     kRAPParams.filterSize, kRAPParams.filterSize);

  for (int i = 0; i <= 2; ++i) {
    for (int j = 0; j <= 2; ++j) {
      for (int k = 0; k <= 2; ++k) {
        for (int l = 0; l <= 2; ++l) {
          // The output math only works out if the stride is 1 or a
          // stride that works for even division
          for (auto rowStride : {1L, kRAPParams.filterStride}) {
            for (auto colStride : {1L, kRAPParams.filterStride}) {
              ReferenceAsymmetricPaddingTest(i * rowStride,
                                             j * rowStride,
                                             k * colStride,
                                             l * colStride).
                testUpdateOutput(input, filters,
                                 rowStride, colStride, folly::none);
            }
          }
        }
      }
    }
  }
}

TEST(CudaConvolutionTest, ReferenceAsymmetricPadding_updateGradInput) {
  auto filters =
    makeFilterTensor(kRAPParams.outputPlanes,
                     kRAPParams.inputPlanes,
                     kRAPParams.filterSize,
                     kRAPParams.filterSize);

  // The output math only works out if the stride is 1 or a
  // stride that works for even division
  for (auto rowStride : {1L, kRAPParams.filterStride}) {
    for (auto colStride : {1L, kRAPParams.filterStride}) {

      auto output =
        makeOutputTensor(
          kRAPParams.batch, kRAPParams.outputPlanes,
          getValidConvSize(kRAPParams.inputRow,
                           kRAPParams.filterSize, rowStride),
          getValidConvSize(kRAPParams.inputCol,
                           kRAPParams.filterSize, colStride));
      for (int i = 0; i <= 2; ++i) {
        for (int j = 0; j <= 2; ++j) {
          for (int k = 0; k <= 2; ++k) {
            for (int l = 0; l <= 2; ++l) {

              ReferenceAsymmetricPaddingTest(i * rowStride,
                                             j * rowStride,
                                             k * colStride,
                                             l * colStride).
                testUpdateGradInput(output, filters,
                                    rowStride, colStride, folly::none);
            }
          }
        }
      }
    }
  }
}

TEST(CudaConvolutionTest, ReferenceAsymmetricPadding_accGradParameters) {
  auto input =
    makeInputTensor(kRAPParams.batch, kRAPParams.inputPlanes,
                    kRAPParams.inputRow, kRAPParams.inputCol);

  // The output math only works out if the stride is 1 or a
  // stride that works for even division
  for (auto rowStride : {1L, kRAPParams.filterStride}) {
    for (auto colStride : {1L, kRAPParams.filterStride}) {

      auto output =
        makeOutputTensor(
          kRAPParams.batch, kRAPParams.outputPlanes,
          getValidConvSize(kRAPParams.inputRow,
                           kRAPParams.filterSize, rowStride),
          getValidConvSize(kRAPParams.inputCol,
                           kRAPParams.filterSize, colStride));

      for (int i = 0; i <= 2; ++i) {
        for (int j = 0; j <= 2; ++j) {
          for (int k = 0; k <= 2; ++k) {
            for (int l = 0; l <= 2; ++l) {

              // Scale is implemented, so test it
              ReferenceAsymmetricPaddingTest(i * rowStride,
                                             j * rowStride,
                                             k * colStride,
                                             l * colStride).
                testAccGradParameters(input, output,
                                      rowStride, colStride,
                                      kAccGradBias, folly::none);
            }
          }
        }
      }
    }
  }
}

//
// InputCentric tests
//

// Constants for the rectangular test cases
constexpr long kInputCentricNumInBatchRect = 4;
constexpr long kInputCentricNumInputPlanesRect = 3;
// kInputCentricNumInputPlanesRegisterRect must be an even divisor of
// kInputCentricNumInputPlanesRect for full unrolling
constexpr long kInputCentricNumInputPlanesRegisterRect = 3;
constexpr long kInputCentricNumFiltersRect = 6;
constexpr long kInputCentricInputRowSizeRect = 32;
constexpr long kInputCentricInputColSizeRect = 48;
constexpr long kInputCentricFilterRowSizeRect = 8;
constexpr long kInputCentricFilterColSizeRect = 4;
constexpr long kInputCentricFilterRowStrideRect = 1;
constexpr long kInputCentricFilterColStrideRect = 2;

TEST(CudaConvolutionTest, ReferenceInputCentric_updateOutput) {
  // Keep things manageable or else too expensive
  long actualNumInBatch = (FLAGS_verify) ?
    std::min((long)4, kInputCentricNumInBatchRect) :
    kInputCentricNumInBatchRect;
  long actualNumFilters = (FLAGS_verify) ?
    std::min((long)6, kInputCentricNumFiltersRect) :
    kInputCentricNumFiltersRect;

  auto input = makeInputTensor(actualNumInBatch,
                               kInputCentricNumInputPlanesRect,
                               kInputCentricInputRowSizeRect,
                               kInputCentricInputColSizeRect);
  auto filters = makeFilterTensor(actualNumFilters,
                                  kInputCentricNumInputPlanesRect,
                                  kInputCentricFilterRowSizeRect,
                                  kInputCentricFilterColSizeRect);

  ReferenceInputCentricTest().
    testUpdateOutput(input,
                     filters,
                     kInputCentricFilterRowStrideRect,
                     kInputCentricFilterColStrideRect,
                     folly::none);
}

TEST(CudaConvolutionTest, InputCentricRelayout_updateOutput) {
  // Keep things manageable or else too expensive
  long actualNumInBatch = (FLAGS_verify) ?
    std::min((long)4, kInputCentricNumInBatchRect) :
    kInputCentricNumInBatchRect;
  long actualNumFilters = (FLAGS_verify) ?
    std::min((long)6, kInputCentricNumFiltersRect) :
    kInputCentricNumFiltersRect;

  auto input = makeInputTensor(actualNumInBatch,
                               kInputCentricNumInputPlanesRect,
                               kInputCentricInputRowSizeRect,
                               kInputCentricInputColSizeRect);
  auto filters = makeFilterTensor(actualNumFilters,
                                  kInputCentricNumInputPlanesRect,
                                  kInputCentricFilterRowSizeRect,
                                  kInputCentricFilterColSizeRect);
  InputCentricTest().testUpdateOutput(input,
                                      filters,
                                      kInputCentricFilterRowStrideRect,
                                      kInputCentricFilterColStrideRect,
                                      folly::none);
}

//
// TorchNN tensor convolution tests
//

constexpr Params kNNParams(16, 3, 8, 8, 16, 4, 2);

TEST(CudaConvolutionTest, TorchNN_updateOutput) {
  auto input =
    makeInputTensor(kNNParams.batch, kNNParams.inputPlanes,
                    kNNParams.inputRow, kNNParams.inputCol);
  auto filters =
    makeFilterTensor(kNNParams.outputPlanes, kNNParams.inputPlanes,
                     kNNParams.filterSize, kNNParams.filterSize);

  TorchTest().testUpdateOutput(input, filters,
                               kNNParams.filterStride,
                               kNNParams.filterStride, folly::none);
}

TEST(CudaConvolutionTest, TorchNN_updateGradInput) {
  auto output =
    makeOutputTensor(kNNParams.batch, kNNParams.outputPlanes,
                     kNNParams.outputRow(), kNNParams.outputCol());

  auto filters =
    makeFilterTensor(kNNParams.outputPlanes, kNNParams.inputPlanes,
                     kNNParams.filterSize, kNNParams.filterSize);

  TorchTest().
    testUpdateGradInput(output, filters,
                        kNNParams.filterStride,
                        kNNParams.filterStride, folly::none);
}

TEST(CudaConvolutionTest, TorchNN_accGradParameters) {
  auto input =
    makeInputTensor(kNNParams.batch, kNNParams.inputPlanes,
                    kNNParams.inputRow, kNNParams.inputCol);

  auto output =
    makeOutputTensor(kNNParams.batch, kNNParams.outputPlanes,
                     kNNParams.outputRow(), kNNParams.outputCol());

  // scale is implemented, so test it
  TorchTest().
    testAccGradParameters(input, output,
                          kNNParams.filterStride, kNNParams.filterStride,
                          kAccGradBias, folly::none);
}

//
// CuFFT tests
//

constexpr Params kCuFFTParams(16, 8, 8, 8, 16, 4, 1);

TEST(CudaConvolutionTest, CuFFT_updateOutput_reference) {
  auto input = makeRandomTestTensor(
    {kCuFFTParams.batch, kCuFFTParams.inputPlanes,
        kCuFFTParams.inputRow, kCuFFTParams.inputCol});
  auto filters = makeRandomTestTensor(
    {kCuFFTParams.outputPlanes, kCuFFTParams.inputPlanes,
        kCuFFTParams.filterSize, kCuFFTParams.filterSize});

  CuFFT(CuFFT::Implementation::Reference).
    testUpdateOutput(input,
                     filters,
                     kCuFFTParams.filterStride,
                     kCuFFTParams.filterStride,
                     folly::none);
}

TEST(CudaConvolutionTest, CuFFT_accGradParameters_reference) {
  auto input = makeRandomTestTensor({
      kCuFFTParams.batch, kCuFFTParams.inputPlanes,
        kCuFFTParams.inputRow, kCuFFTParams.inputCol});

  auto output = makeRandomTestTensor({
      kCuFFTParams.batch, kCuFFTParams.outputPlanes,
        kCuFFTParams.outputRow(), kCuFFTParams.outputCol()});

  CuFFT(CuFFT::Implementation::Reference).
    testAccGradParameters(input,
                          output,
                          kCuFFTParams.filterStride,
                          kCuFFTParams.filterStride,
                          kAccGradBias,
                          folly::none);
}

TEST(CudaConvolutionTest, CuFFT_updateGradInput_reference) {
  auto output =
    makeRandomTestTensor({
        kCuFFTParams.batch, kCuFFTParams.outputPlanes,
        kCuFFTParams.outputRow(), kCuFFTParams.outputCol()});

  auto filters = makeRandomTestTensor(
    {kCuFFTParams.outputPlanes, kCuFFTParams.inputPlanes,
        kCuFFTParams.filterSize, kCuFFTParams.filterSize});

  CuFFT(CuFFT::Implementation::Reference).
    testUpdateGradInput(output,
                        filters,
                        kCuFFTParams.filterStride,
                        kCuFFTParams.filterStride,
                        folly::none);
}

TEST(CudaConvolutionTest, CuFFT_updateOutput) {
  auto input = makeRandomTestTensor(
    {kCuFFTParams.batch, kCuFFTParams.inputPlanes,
        kCuFFTParams.inputRow, kCuFFTParams.inputCol});
  auto filters = makeRandomTestTensor(
    {kCuFFTParams.outputPlanes, kCuFFTParams.inputPlanes,
        kCuFFTParams.filterSize, kCuFFTParams.filterSize});

  CuFFT().testUpdateOutput(input,
                           filters,
                           kCuFFTParams.filterStride,
                           kCuFFTParams.filterStride,
                           folly::none);
}

TEST(CudaConvolutionTest, CuFFT_accGradParameters) {
  auto input = makeRandomTestTensor({
      kCuFFTParams.batch, kCuFFTParams.inputPlanes,
        kCuFFTParams.inputRow, kCuFFTParams.inputCol});

  auto output = makeRandomTestTensor({
      kCuFFTParams.batch, kCuFFTParams.outputPlanes,
        kCuFFTParams.outputRow(), kCuFFTParams.outputCol()});

  CuFFT().testAccGradParameters(input,
                                output,
                                kCuFFTParams.filterStride,
                                kCuFFTParams.filterStride,
                                kAccGradBias,
                                folly::none);
}

TEST(CudaConvolutionTest, CuFFT_updateGradInput) {
  auto output =
    makeRandomTestTensor({
        kCuFFTParams.batch, kCuFFTParams.outputPlanes,
          kCuFFTParams.outputRow(), kCuFFTParams.outputCol()});

  auto filters = makeRandomTestTensor(
    {kCuFFTParams.outputPlanes, kCuFFTParams.inputPlanes,
        kCuFFTParams.filterSize, kCuFFTParams.filterSize});

  CuFFT().testUpdateGradInput(output,
                              filters,
                              kCuFFTParams.filterStride,
                              kCuFFTParams.filterStride,
                              folly::none);
}

TEST(CudaConvolutionTest, CuFFT_updateGradInput_fixed) {
  constexpr int outSize = 9;
  constexpr int kerSize = 3;

  auto output =
    makeTestTensor({1, 1, getValidConvSize(outSize, kerSize, 1),
          getValidConvSize(outSize, kerSize, 1)}, 1.0f);

  auto filters = makeTestTensor({1, 1, kerSize, kerSize}, 1.0f);

  auto cufft = CuFFT();
  cufft.testUpdateGradInput(output, filters,
                            kCuFFTParams.filterStride,
                            kCuFFTParams.filterStride, folly::none);

  // Check borders with a well-known kernel, expected:
  // 1 2 3 3 3 3 3 2 1
  // 2 4 6 6 6 6 6 4 2
  // 3 6 9 9 9 9 9 6 3
  // 3 6 9 9 9 9 9 6 3
  // 3 6 9 9 9 9 9 6 3
  // 3 6 9 9 9 9 9 6 3
  // 3 6 9 9 9 9 9 6 3
  // 2 4 6 6 6 6 6 4 2
  // 1 2 3 3 3 3 3 2 1
  auto expectedInput = makeTestTensor({1, 1, outSize, outSize}, (float)outSize);
  for (int r = 0; r < outSize; ++r) {
    expectedInput.at({0, 0, r, 0}) = 3.0f;
    expectedInput.at({0, 0, r, 1}) = 6.0f;
    expectedInput.at({0, 0, r, outSize - 2}) = 6.0f;
    expectedInput.at({0, 0, r, outSize - 1}) = 3.0f;
  }
  for (int c = 0; c < outSize; ++c) {
    expectedInput.at({0, 0, 0, c}) = 3.0f;
    expectedInput.at({0, 0, 1, c}) = 6.0f;
    expectedInput.at({0, 0, outSize - 2, c}) = 6.0f;
    expectedInput.at({0, 0, outSize - 1, c}) = 3.0f;
  }
  expectedInput.at({0, 0, 0, 0}) = 1.0f;
  expectedInput.at({0, 0, 0, outSize - 1}) = 1.0f;
  expectedInput.at({0, 0, outSize - 1, 0}) = 1.0f;
  expectedInput.at({0, 0, outSize - 1, outSize - 1}) = 1.0f;
  expectedInput.at({0, 0, 1, 1}) = 4.0f;
  expectedInput.at({0, 0, 1, outSize - 2}) = 4.0f;
  expectedInput.at({0, 0, outSize - 2, 1}) = 4.0f;
  expectedInput.at({0, 0, outSize - 2, outSize - 2}) = 4.0f;
  expectedInput.at({0, 0, 0, outSize - 2}) = 2.0f;
  expectedInput.at({0, 0, 1, outSize - 1}) = 2.0f;
  expectedInput.at({0, 0, outSize - 2, outSize - 1}) = 2.0f;
  expectedInput.at({0, 0, outSize - 1, outSize - 2}) = 2.0f;
  expectedInput.at({0, 0, 0, 1}) = 2.0f;
  expectedInput.at({0, 0, 1, 0}) = 2.0f;
  expectedInput.at({0, 0, outSize - 2, 0}) = 2.0f;
  expectedInput.at({0, 0, outSize - 1, 1}) = 2.0f;

  CuFFT::checkExpectedInput(
    expectedInput,
    copyFromCuda(nullptr, cufft.saveInputTHCudaTensor.get()));
}

} } } } // namespace

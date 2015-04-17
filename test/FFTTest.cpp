// Copyright 2004-present Facebook. All Rights Reserved.

#include "torch/fb/fbcunn/src/DeviceTensorUtils.h"
#include "THCTensor.h"
#include "torch/fb/fbcunn/src/fft/CuFFTWrapper.cuh"
#include "torch/fb/fbcunn/test/InputCentricConvolution_UpdateOutput.cuh"
#include "torch/fb/fbcunn/test/ReferenceConvolutions.h"
#include "torch/fb/fbcunn/test/TestUtils.h"

#include <folly/Optional.h>

#include <glog/logging.h>
#include <gtest/gtest.h>

using namespace std;
using namespace facebook::cuda;
using namespace facebook::deeplearning::torch;

DEFINE_bool(verify, true, "Run the convolution and verify the output");

// Override gtest_main so as to parse the --verify flag
int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  google::ParseCommandLineFlags(&argc, &argv, true);
  return RUN_ALL_TESTS();
}

namespace facebook { namespace deeplearning { namespace torch { namespace test {

enum class DataValueSpecification : bool {
  Constant = true,
  Variable = false
};

struct TestConfig {
  explicit TestConfig(long batch = 2,
                      long fftd = 2,
                      FFTOutputSpecification fos =
                      FFTOutputSpecification::OutOfPlace,
                      DataValueSpecification cvs =
                      DataValueSpecification::Constant) :
      batchDim(batch),
      fftDim(fftd),
      inPlace(fos),
      constantTensor(cvs) {}

  long batchDim;
  long fftDim;
  FFTOutputSpecification inPlace;
  DataValueSpecification constantTensor;
};


// At the moment only really supports BatchDim = 2 and FFTDim = 2
// TODO:#4846735 extend this
// Always dim 4 (3b+1fft, 2b+2fft, 1b+3fft) atm, extend later
template <long BatchDim = 2, long FFTDim = 2>
class FFTTestBase : public ::testing::Test {
 private:
  void setSize(initializer_list<long> sizeInputTensor,
               initializer_list<long> paddedFFTSize) {
    CHECK_EQ(BatchDim + FFTDim, 4); // extend this
    CHECK_EQ(BatchDim + FFTDim, sizeInputTensor.size());
    CHECK_EQ(FFTDim, paddedFFTSize.size());
    tensorSize = sizeInputTensor;
    FFTSize = paddedFFTSize;

    input = (cfg.constantTensor == DataValueSpecification::Constant) ?
      makeTestTensor(sizeInputTensor, 1.0f) : makeTestTensor(sizeInputTensor);
  }

 public:
  void configure(TestConfig config,
                 initializer_list<long> sizeInputTensor,
                 initializer_list<long> paddedFFTSize,
                 folly::Optional<Tensor<float>> expected = folly::none) {
    cfg = config;
    setSize(sizeInputTensor, paddedFFTSize);
    if (expected) {
      expectedComplex = *expected;
    }

    auto realComplexPair =
      makeCuFFTTensors<FFTDim>(nullptr, input, FFTSize, cfg.inPlace);
    inputTHCudaTensor = std::move(realComplexPair.first);
    fftTHCudaTensor = std::move(realComplexPair.second);

    inputCudaTensor =
      torchToDeviceTensor<float, FFTDim + BatchDim>(
        nullptr, inputTHCudaTensor.get());
    outputCudaTensor =
      torchToDeviceTensor<float, FFTDim + BatchDim + 1>(
        nullptr, fftTHCudaTensor.get());

    if (cfg.inPlace == FFTOutputSpecification::InPlace) {
      CHECK_EQ(inputCudaTensor.data(), outputCudaTensor.data());
    } else {
      CHECK_NE(inputCudaTensor.data(), outputCudaTensor.data());
    }

    // Check inputTHCudaTensor properly sized
    for (int i = 0; i < BatchDim; ++i) {
      CHECK_EQ(tensorSize[i], inputCudaTensor.getSize(i));
    }
    for (int i = 0; i < FFTDim - 1; ++i) {
      CHECK_EQ(FFTSize[i], inputCudaTensor.getSize(BatchDim + i)) <<
        " i = " << i;
    }
    // Check inputTHCudaTensor properly strided
    const long kHermitianColSize = numFFTCols(FFTSize[FFTDim - 1]);
    if (config.inPlace == FFTOutputSpecification::InPlace) {
      CHECK_EQ(2 * kHermitianColSize,
               inputCudaTensor.getStride(BatchDim + FFTDim - 2));
    } else {
      CHECK_EQ(FFTSize[FFTDim - 1],
               inputCudaTensor.getStride(BatchDim + FFTDim - 2));
    }
    CHECK_EQ(1, inputCudaTensor.getStride(BatchDim + FFTDim - 1));

    // Check fftTHCudaTensor properly sized
    for (int i = 0; i < BatchDim; ++i) {
      CHECK_EQ(tensorSize[i], outputCudaTensor.getSize(i));
    }
    for (int i = 0; i < FFTDim - 2; ++i) {
      CHECK_EQ(FFTSize[i], outputCudaTensor.getSize(BatchDim + i));
    }
    CHECK_EQ(kHermitianColSize,
             outputCudaTensor.getSize((BatchDim + FFTDim + 1) - 2));
    CHECK_EQ(2, outputCudaTensor.getSize((BatchDim + FFTDim + 1) - 1));

    // Check fftTHCudaTensor properly strided
    CHECK_EQ(kHermitianColSize * 2,
             outputCudaTensor.getStride((BatchDim + FFTDim + 1) - 3));
    CHECK_EQ(2, outputCudaTensor.getStride((BatchDim + FFTDim + 1) - 2));
    CHECK_EQ(1, outputCudaTensor.getStride((BatchDim + FFTDim + 1) - 1));
  }

  void checkExpectedOutput(Tensor<float> expectedComplex,
                           Tensor<float> actualComplex,
                           float relativeError = 1e-6f) {
    CHECK_EQ(5, actualComplex.ndims());
    CHECK_EQ(5, expectedComplex.ndims());
    for (int i = 0; i < expectedComplex.size(0); ++i) {
      Tensor<float> tmp1(expectedComplex, Tensor<float>::UNIQUE);
      tmp1.select(0, i);
      Tensor<float> tmp2(actualComplex, Tensor<float>::UNIQUE);
      tmp2.select(0, i);
      auto comparison =
        compareTensors(tmp1, tmp2, relativeError, 10, true);
      ASSERT_TRUE(comparison.first) <<
        "Mismatch on i = " << i<< ": " << comparison.second;
    }
  }

  void checkExpectedInput(Tensor<float> expectedInput,
                          Tensor<float> actualInput,
                          float relativeError = 1e-6f) {
    CHECK_EQ(4, actualInput.ndims());
    CHECK_EQ(4, expectedInput.ndims());
    auto comparison =
      compareTensors(expectedInput, actualInput, relativeError, 10, true);
    ASSERT_TRUE(comparison.first) << "Mismatch " << comparison.second;
  }

  TestConfig cfg;

  // 2-D + (batch, filters, row, col)
  vector<long> tensorSize;

  // If 3-D (planes, row, col)
  // If 2-D (row, col)
  // If 1-D (col)
  vector<long> FFTSize;

  // Input
  Tensor<float> input;
  // Expected complex output
  Tensor<float> expectedComplex;

  // Cuda side
  unique_ptr<THCudaTensor, CudaTensorDeleter> inputTHCudaTensor;
  unique_ptr<THCudaTensor, CudaTensorDeleter> fftTHCudaTensor;

  DeviceTensor<float, BatchDim + FFTDim> inputCudaTensor;
  DeviceTensor<float, BatchDim + FFTDim + 1> outputCudaTensor;
};

class FFT1DTest : public FFTTestBase<3, 1> { };
class FFT2DTest : public FFTTestBase<2, 2> { };
class FFT3DTest : public FFTTestBase<1, 3> { };

TEST_F(FFT2DTest, test2x2ConstantInPlace) {
  constexpr int kBatchDims = 2;
  constexpr int kDims = 2;
  constexpr FFTOutputSpecification kInPlace = FFTOutputSpecification::InPlace;

  TestConfig cfg(kBatchDims, kDims, kInPlace);

  constexpr int kNumBatches = 1;
  constexpr int kNumInputPlanes = 1;
  constexpr int kNumRows = 2;
  constexpr int kNumPaddedRows = 2;
  constexpr int kNumCols = 2;
  constexpr int kNumPaddedCols = 2;

  Tensor<float> expected(
    { kNumBatches,
        kNumInputPlanes,
        kNumRows,
        numFFTCols(kNumCols),
        2 });
  expected.fill(0.0f);
  expected.at({0, 0, 0, 0, 0}) = 4.0f;

  configure(cfg,
            {1, 1, kNumRows, kNumCols},
            {kNumPaddedRows, kNumPaddedCols},
            expected);

  fft2d<2>(inputCudaTensor, outputCudaTensor);

  checkExpectedOutput(expected,
                      copyFromCuda(nullptr, fftTHCudaTensor.get()));
}

TEST_F(FFT2DTest, test2x2ConstantOutOfPlace) {
  constexpr int kBatchDims = 2;
  constexpr int kDims = 2;

  TestConfig cfg(kBatchDims, kDims);

  constexpr int kNumBatches = 1;
  constexpr int kNumInputPlanes = 1;
  constexpr int kNumRows = 2;
  constexpr int kNumPaddedRows = 2;
  constexpr int kNumCols = 2;
  constexpr int kNumPaddedCols = 2;

  Tensor<float> expected(
    { kNumBatches,
        kNumInputPlanes,
        kNumRows,
        numFFTCols(kNumCols),
        2 });
  expected.fill(0.0f);
  expected.at({0, 0, 0, 0, 0}) = 4.0f;

  configure(cfg,
            {1, 1, kNumRows, kNumCols},
            {kNumPaddedRows, kNumPaddedCols},
            expected);

  fft2d<2>(inputCudaTensor, outputCudaTensor);

  checkExpectedOutput(expected,
                      copyFromCuda(nullptr, fftTHCudaTensor.get()));
}

TEST_F(FFT2DTest, test2x2VariableInPlace) {
  constexpr int kBatchDims = 2;
  constexpr int kDims = 2;
  constexpr FFTOutputSpecification kInPlace = FFTOutputSpecification::InPlace;
  constexpr DataValueSpecification kConstant = DataValueSpecification::Variable;

  TestConfig cfg(kBatchDims, kDims, kInPlace, kConstant);

  constexpr int kNumBatches = 1;
  constexpr int kNumInputPlanes = 1;
  constexpr int kNumRows = 2;
  constexpr int kNumPaddedRows = 2;
  constexpr int kNumCols = 2;
  constexpr int kNumPaddedCols = 2;

  Tensor<float> expected(
    { kNumBatches,
        kNumInputPlanes,
        kNumRows,
        numFFTCols(kNumCols),
        2 });
  expected.fill(0.0f);
  expected.at({0, 0, 0, 0, 0}) =  1.4f;
  expected.at({0, 0, 0, 1, 0}) = -0.8f;
  expected.at({0, 0, 1, 0, 0}) = -0.6f;

  configure(cfg,
            {1, 1, kNumRows, kNumCols},
            {kNumPaddedRows, kNumPaddedCols},
            expected);

  fft2d<2>(inputCudaTensor, outputCudaTensor);

  checkExpectedOutput(expected,
                      copyFromCuda(nullptr, fftTHCudaTensor.get()));
}

TEST_F(FFT2DTest, test2x2VariableOutOfPlace) {
  constexpr int kBatchDims = 2;
  constexpr int kDims = 2;
  constexpr FFTOutputSpecification kInPlace =
    FFTOutputSpecification::OutOfPlace;
  constexpr DataValueSpecification kConstant = DataValueSpecification::Variable;

  TestConfig cfg(kBatchDims, kDims, kInPlace, kConstant);

  constexpr int kNumBatches = 1;
  constexpr int kNumInputPlanes = 1;
  constexpr int kNumRows = 2;
  constexpr int kNumPaddedRows = 2;
  constexpr int kNumCols = 2;
  constexpr int kNumPaddedCols = 2;

  Tensor<float> expected(
    { kNumBatches,
        kNumInputPlanes,
        kNumRows,
        numFFTCols(kNumCols),
        2 });
  expected.fill(0.0f);
  expected.at({0, 0, 0, 0, 0}) =  1.4f;
  expected.at({0, 0, 0, 1, 0}) = -0.8f;
  expected.at({0, 0, 1, 0, 0}) = -0.6f;

  configure(cfg,
            {1, 1, kNumRows, kNumCols},
            {kNumPaddedRows, kNumPaddedCols},
            expected);

  fft2d<2>(inputCudaTensor, outputCudaTensor);

  checkExpectedOutput(expected,
                      copyFromCuda(nullptr, fftTHCudaTensor.get()));
}

TEST_F(FFT2DTest, test1x2ConstantInPlacePadded) {
  constexpr int kBatchDims = 2;
  constexpr int kDims = 2;
  constexpr FFTOutputSpecification kInPlace = FFTOutputSpecification::InPlace;

  TestConfig cfg(kBatchDims, kDims, kInPlace);

  constexpr int kNumBatches = 1;
  constexpr int kNumInputPlanes = 1;
  constexpr int kNumRows = 1;
  constexpr int kNumPaddedRows = 4;
  constexpr int kNumCols = 2;
  constexpr int kNumPaddedCols = 2;

  Tensor<float> expected(
    { kNumBatches,
        kNumInputPlanes,
        kNumPaddedRows,
        numFFTCols(kNumPaddedCols),
        2 });
  expected.fill(0.0f);
  expected.at({0, 0, 0, 0, 0}) = 2.0f;
  expected.at({0, 0, 1, 0, 0}) = 2.0f;
  expected.at({0, 0, 2, 0, 0}) = 2.0f;
  expected.at({0, 0, 3, 0, 0}) = 2.0f;

  configure(cfg,
            {1, 1, kNumRows, kNumCols},
            {kNumPaddedRows, kNumPaddedCols},
            expected);

  fft2d<2>(inputCudaTensor, outputCudaTensor);

  checkExpectedOutput(expected,
                      copyFromCuda(nullptr, fftTHCudaTensor.get()));
}

TEST_F(FFT2DTest, test1x2ConstantOutOfPlacePadded) {
  constexpr int kBatchDims = 2;
  constexpr int kDims = 2;

  TestConfig cfg(kBatchDims, kDims);

  constexpr int kNumBatches = 1;
  constexpr int kNumInputPlanes = 1;
  constexpr int kNumRows = 1;
  constexpr int kNumPaddedRows = 4;
  constexpr int kNumCols = 2;
  constexpr int kNumPaddedCols = 2;

  Tensor<float> expected(
    { kNumBatches,
        kNumInputPlanes,
        kNumPaddedRows,
        numFFTCols(kNumPaddedCols),
        2 });
  expected.fill(0.0f);

  expected.at({0, 0, 0, 0, 0}) = 2.0f;
  expected.at({0, 0, 1, 0, 0}) = 2.0f;
  expected.at({0, 0, 2, 0, 0}) = 2.0f;
  expected.at({0, 0, 3, 0, 0}) = 2.0f;

  configure(cfg,
            {1, 1, kNumRows, kNumCols},
            {kNumPaddedRows, kNumPaddedCols},
            expected);

  fft2d<2>(inputCudaTensor, outputCudaTensor);

  checkExpectedOutput(expected,
                      copyFromCuda(nullptr, fftTHCudaTensor.get()));
}

TEST_F(FFT2DTest, test2x2ConstantInPlacePadded) {
  constexpr int kBatchDims = 2;
  constexpr int kDims = 2;
  constexpr FFTOutputSpecification kInPlace = FFTOutputSpecification::InPlace;

  TestConfig cfg(kBatchDims, kDims, kInPlace);

  constexpr int kNumBatches = 1;
  constexpr int kNumInputPlanes = 1;
  constexpr int kNumRows = 2;
  constexpr int kNumPaddedRows = 3;
  constexpr int kNumCols = 2;
  constexpr int kNumPaddedCols = 4;

  Tensor<float> expected(
    { kNumBatches,
        kNumInputPlanes,
        kNumPaddedRows,
        numFFTCols(kNumPaddedCols),
        2 });
  expected.fill(0.0f);

  expected.at({0, 0, 0, 0, 0}) =  4.0f;
  expected.at({0, 0, 0, 1, 0}) =  2.0f;
  expected.at({0, 0, 0, 1, 1}) = -2.0f;
  expected.at({0, 0, 0, 2, 0}) =  0.0f;
  expected.at({0, 0, 0, 2, 1}) =  0.0f;

  expected.at({0, 0, 1, 0, 0}) =  1.0f;
  expected.at({0, 0, 1, 0, 1}) = -1.732051f;
  expected.at({0, 0, 1, 1, 0}) = -0.366025f;
  expected.at({0, 0, 1, 1, 1}) = -1.366025f;
  expected.at({0, 0, 1, 2, 0}) =  0.0f;
  expected.at({0, 0, 1, 2, 1}) =  0.0f;

  expected.at({0, 0, 2, 0, 0}) =  1.0f;
  expected.at({0, 0, 2, 0, 1}) =  1.732051f;
  expected.at({0, 0, 2, 1, 0}) =  1.366025f;
  expected.at({0, 0, 2, 1, 1}) =  0.366025f;
  expected.at({0, 0, 2, 2, 0}) =  0.0f;
  expected.at({0, 0, 2, 2, 1}) =  0.0f;

  configure(cfg,
            {1, 1, kNumRows, kNumCols},
            {kNumPaddedRows, kNumPaddedCols},
            expected);

  fft2d<2>(inputCudaTensor, outputCudaTensor);

  // One element does not check at 1e-6f error
  checkExpectedOutput(expected,
                      copyFromCuda(nullptr, fftTHCudaTensor.get()),
                      5e-5f);
}

TEST_F(FFT2DTest, test2x2ConstantOutOfPlacePadded) {
  constexpr int kBatchDims = 2;
  constexpr int kDims = 2;

  TestConfig cfg(kBatchDims, kDims);

  constexpr int kNumBatches = 3;
  constexpr int kNumInputPlanes = 5;
  constexpr int kNumRows = 2;
  constexpr int kNumPaddedRows = 4;
  constexpr int kNumCols = 2;
  constexpr int kNumPaddedCols = 3;

  Tensor<float> expected(
    { kNumBatches,
        kNumInputPlanes,
        kNumPaddedRows,
        numFFTCols(kNumPaddedCols),
        2 });
  expected.fill(0.0f);

  for (int batch = 0; batch < kNumBatches; ++batch) {
    for (int outputPlane = 0; outputPlane < kNumInputPlanes; ++outputPlane) {
      expected.at({batch, outputPlane, 0, 0, 0}) =  4.0f;
      expected.at({batch, outputPlane, 0, 1, 0}) =  1.0f;
      expected.at({batch, outputPlane, 0, 1, 1}) = -1.732051f;
      // 1.0f; Hermitian symmetry
      // 1.732051f; Hermitian symmetry

      expected.at({batch, outputPlane, 1, 0, 0}) =  2.0f;
      expected.at({batch, outputPlane, 1, 0, 1}) = -2.0f;
      expected.at({batch, outputPlane, 1, 1, 0}) = -0.366025f;
      expected.at({batch, outputPlane, 1, 1, 1}) = -1.366025f;
      // 0.366025f; Hermitian symmetry
      // 1.366025f; Hermitian symmetry

      // Third row full 0

      expected.at({batch, outputPlane, 3, 0, 0}) =  2.0f;
      expected.at({batch, outputPlane, 3, 0, 1}) =  2.0f;
      expected.at({batch, outputPlane, 3, 1, 0}) =  1.366025f;
      expected.at({batch, outputPlane, 3, 1, 1}) = -0.366025f;
      // -0.366025f; Hermitian symmetry
      // -1.366025f; Hermitian symmetry
    }
  }
  configure(cfg,
            {kNumBatches, kNumInputPlanes, kNumRows, kNumCols},
            {kNumPaddedRows, kNumPaddedCols},
            expected);

  fft2d<2>(inputCudaTensor, outputCudaTensor);

  // One element does not check at 1e-6f error
  checkExpectedOutput(expected,
                      copyFromCuda(nullptr, fftTHCudaTensor.get()),
                      5e-5f);
}

TEST_F(FFT2DTest, testInverseOutOfPlace) {
  constexpr int kBatchDims = 2;
  constexpr int kDims = 2;
  constexpr FFTOutputSpecification kInPlace =
    FFTOutputSpecification::OutOfPlace;
  constexpr DataValueSpecification kConstant = DataValueSpecification::Variable;
  TestConfig cfg(kBatchDims, kDims, kInPlace, kConstant);

  constexpr int kNumBatches = 4;
  constexpr int kNumInputPlanes = 7;
  constexpr int kNumRows = 15;
  constexpr int kNumPaddedRows = 15;
  constexpr int kNumCols = 33;
  constexpr int kNumPaddedCols = 33;

  Tensor<float> expected(
    { kNumBatches,
        kNumInputPlanes,
        kNumPaddedRows,
        numFFTCols(kNumPaddedCols),
        2 });
  expected.fill(0.0f);

  configure(cfg,
            {kNumBatches, kNumInputPlanes, kNumRows, kNumCols},
            {kNumPaddedRows, kNumPaddedCols},
            expected);

  fft2d<2>(inputCudaTensor, outputCudaTensor);
  fft2d<2>(inputCudaTensor, outputCudaTensor, FFTParameters().inverse());

  // First element does not check at 5e-5f error
  checkExpectedInput(input,
                     copyFromCuda(nullptr, inputTHCudaTensor.get()),
                     5e-4f);
}

TEST_F(FFT2DTest, testInverseInPlace) {
  constexpr int kBatchDims = 2;
  constexpr int kDims = 2;
  constexpr FFTOutputSpecification kInPlace = FFTOutputSpecification::InPlace;
  constexpr DataValueSpecification kConstant = DataValueSpecification::Variable;
  TestConfig cfg(kBatchDims, kDims, kInPlace, kConstant);

  constexpr int kNumBatches = 4;
  constexpr int kNumInputPlanes = 7;
  constexpr int kNumRows = 16;
  constexpr int kNumPaddedRows = 16;
  constexpr int kNumCols = 7;
  constexpr int kNumPaddedCols = 7;

  Tensor<float> expected(
    { kNumBatches,
        kNumInputPlanes,
        kNumPaddedRows,
        numFFTCols(kNumPaddedCols),
        2 });
  expected.fill(0.0f);

  configure(cfg,
            {kNumBatches, kNumInputPlanes, kNumRows, kNumCols},
            {kNumPaddedRows, kNumPaddedCols},
            expected);

  fft2d<2>(inputCudaTensor, outputCudaTensor);
  fft2d<2>(inputCudaTensor, outputCudaTensor, FFTParameters().inverse());

  // First element does not check at 1e-6f error
  checkExpectedInput(input,
                     copyFromCuda(nullptr, inputTHCudaTensor.get()),
                     5e-5f);
}

TEST_F(FFT2DTest, testInverseOutOfPlacePadded) {
  constexpr int kBatchDims = 2;
  constexpr int kDims = 2;
  constexpr FFTOutputSpecification kInPlace =
    FFTOutputSpecification::OutOfPlace;
  constexpr DataValueSpecification kConstant = DataValueSpecification::Variable;
  TestConfig cfg(kBatchDims, kDims, kInPlace, kConstant);

  constexpr int kNumBatches = 4;
  constexpr int kNumInputPlanes = 7;
  constexpr int kNumRows = 12;
  constexpr int kNumPaddedRows = 15;
  constexpr int kNumCols = 30;
  constexpr int kNumPaddedCols = 33;

  Tensor<float> expected(
    { kNumBatches,
        kNumInputPlanes,
        kNumPaddedRows,
        numFFTCols(kNumPaddedCols),
        2 });
  expected.fill(0.0f);

  configure(cfg,
            {kNumBatches, kNumInputPlanes, kNumRows, kNumCols},
            {kNumPaddedRows, kNumPaddedCols},
            expected);

  fft2d<2>(inputCudaTensor, outputCudaTensor);
  fft2d<2>(inputCudaTensor, outputCudaTensor, FFTParameters().inverse());

  // First element does not check at 5e-5f error
  checkExpectedInput(input,
                     copyFromCuda(nullptr, inputTHCudaTensor.get()),
                     5e-4f);
}

TEST_F(FFT1DTest, test1x4VariableOutOfPlacePadded) {
  constexpr int kBatchDims = 3;
  constexpr int kDims = 1;
  constexpr FFTOutputSpecification kInPlace =
    FFTOutputSpecification::OutOfPlace;
  constexpr DataValueSpecification kConstant = DataValueSpecification::Variable;
  TestConfig cfg(kBatchDims, kDims, kInPlace, kConstant);

  constexpr int kNumBatches = 1;     // Only test 1 unless you want to
                                     // manually insert expected outputs
  constexpr int kNumInputPlanes = 1; // Only test 1 unless you want to
                                     // manually insert expected outputs
  constexpr int kNumRows = 1;        // Acts like a batch dimension
                                     // Only test 1 unless you want to
                                     // manually insert expected outputs
  constexpr int kNumPaddedRows = kNumRows;  // Acts like a batch dimension
  constexpr int kNumCols = 4;
  constexpr int kNumPaddedCols = 8;

  Tensor<float> expected(
    { kNumBatches,
        kNumInputPlanes,
        kNumRows,
        numFFTCols(kNumPaddedCols),
        2 });
  expected.fill(0.0f);

  // Stop at 5 since Hermitian symmetry
  expected.at({0, 0, 0, 0, 0}) =  2.400000f;
  expected.at({0, 0, 0, 1, 0}) = -0.565685f;
  expected.at({0, 0, 0, 2, 0}) = -0.800000f;
  expected.at({0, 0, 0, 3, 0}) =  0.565685f;
  expected.at({0, 0, 0, 4, 0}) = -0.800000f;

  expected.at({0, 0, 0, 0, 1}) =  0.000000f;
  expected.at({0, 0, 0, 1, 1}) = -1.931371f;
  expected.at({0, 0, 0, 2, 1}) =  0.800000f;
  expected.at({0, 0, 0, 3, 1}) = -0.331371f;
  expected.at({0, 0, 0, 4, 1}) =  0.000000f;

  configure(cfg,
            {kNumBatches, kNumInputPlanes, kNumRows, kNumCols},
            {kNumPaddedCols},
            expected);

  fft1d<3>(inputCudaTensor, outputCudaTensor);

  checkExpectedOutput(expected,
                      copyFromCuda(nullptr, fftTHCudaTensor.get()));
}

TEST_F(FFT1DTest, test1x4VariableInPlacePadded) {
  constexpr int kBatchDims = 3;
  constexpr int kDims = 1;
  constexpr FFTOutputSpecification kInPlace = FFTOutputSpecification::InPlace;
  constexpr DataValueSpecification kConstant = DataValueSpecification::Variable;
  TestConfig cfg(kBatchDims, kDims, kInPlace, kConstant);


  constexpr int kNumBatches = 1;     // Only test 1 unless you want to
                                     // manually insert expected outputs
  constexpr int kNumInputPlanes = 1; // Only test 1 unless you want to
                                     // manually insert expected outputs
  constexpr int kNumRows = 1;        // Acts like a batch dimension
                                     // Only test 1 unless you want to
                                     // manually insert expected outputs
  constexpr int kNumPaddedRows = kNumRows;  // Acts like a batch dimension
  constexpr int kNumCols = 4;
  constexpr int kNumPaddedCols = 8;

  Tensor<float> expected(
    { kNumBatches,
        kNumInputPlanes,
        kNumRows,
        numFFTCols(kNumPaddedCols),
        2 });
  expected.fill(0.0f);

  // Stop at 5 since Hermitian symmetry
  expected.at({0, 0, 0, 0, 0}) =  2.400000f;
  expected.at({0, 0, 0, 1, 0}) = -0.565685f;
  expected.at({0, 0, 0, 2, 0}) = -0.800000f;
  expected.at({0, 0, 0, 3, 0}) =  0.565685f;
  expected.at({0, 0, 0, 4, 0}) = -0.800000f;

  expected.at({0, 0, 0, 0, 1}) =  0.000000f;
  expected.at({0, 0, 0, 1, 1}) = -1.931371f;
  expected.at({0, 0, 0, 2, 1}) =  0.800000f;
  expected.at({0, 0, 0, 3, 1}) = -0.331371f;
  expected.at({0, 0, 0, 4, 1}) =  0.000000f;

  configure(cfg,
            {kNumBatches, kNumInputPlanes, kNumRows, kNumCols},
            {kNumPaddedCols},
            expected);

  fft1d<3>(inputCudaTensor, outputCudaTensor);

  checkExpectedOutput(expected,
                      copyFromCuda(nullptr, fftTHCudaTensor.get()));
}

TEST_F(FFT1DTest, testInverseInPlace) {
  constexpr int kBatchDims = 3;
  constexpr int kDims = 1;
  constexpr FFTOutputSpecification kInPlace = FFTOutputSpecification::InPlace;
  constexpr DataValueSpecification kConstant = DataValueSpecification::Variable;
  TestConfig cfg(kBatchDims, kDims, kInPlace, kConstant);

  constexpr int kNumBatches = 4;
  constexpr int kNumInputPlanes = 7;
  constexpr int kNumRows = 16;             // Acts like a batch dim
  constexpr int kNumPaddedRows = kNumRows; // Acts like a batch dim
  constexpr int kNumCols = 7;
  constexpr int kNumPaddedCols = 7;

  Tensor<float> expected(
    { kNumBatches,
        kNumInputPlanes,
        kNumPaddedRows,
        numFFTCols(kNumPaddedCols),
        2 });
  expected.fill(0.0f);

  configure(cfg,
            {kNumBatches, kNumInputPlanes, kNumRows, kNumCols},
            {kNumPaddedCols},
            expected);

  fft1d<3>(inputCudaTensor, outputCudaTensor);
  fft1d<3>(inputCudaTensor, outputCudaTensor, FFTParameters().inverse());

  checkExpectedInput(input,
                     copyFromCuda(nullptr, inputTHCudaTensor.get()));
}

TEST_F(FFT1DTest, testInverseOutOfPlacePadded) {
  constexpr int kBatchDims = 3;
  constexpr int kDims = 1;
  constexpr FFTOutputSpecification kInPlace =
    FFTOutputSpecification::OutOfPlace;
  constexpr DataValueSpecification kConstant = DataValueSpecification::Variable;
  TestConfig cfg(kBatchDims, kDims, kInPlace, kConstant);

  constexpr int kNumBatches = 2;
  constexpr int kNumInputPlanes = 3;
  constexpr int kNumRows = 4;             // Acts like a batch dim
  constexpr int kNumPaddedRows = kNumRows; // Acts like a batch dim
  constexpr int kNumCols = 30;
  constexpr int kNumPaddedCols = 33;

  Tensor<float> expected(
    { kNumBatches,
        kNumInputPlanes,
        kNumPaddedRows,
        numFFTCols(kNumPaddedCols),
        2 });
  expected.fill(0.0f);

  configure(cfg,
            {kNumBatches, kNumInputPlanes, kNumRows, kNumCols},
            {kNumPaddedCols},
            expected);

  fft1d<3>(inputCudaTensor, outputCudaTensor);
  fft1d<3>(inputCudaTensor, outputCudaTensor, FFTParameters().inverse());

  checkExpectedInput(input,
                     copyFromCuda(nullptr, inputTHCudaTensor.get()),
                     5e-5f);
}

} } } } // namespace

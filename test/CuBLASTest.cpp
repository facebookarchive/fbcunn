// Copyright 2004-present Facebook. All Rights Reserved.

#include "torch/fb/fbcunn/src/DeviceTensorUtils.h"
#include "THCTensor.h"
#include "torch/fb/fbcunn/src/CuBLASWrapper.h"
#include "torch/fb/fbcunn/test/TestUtils.h"

#include <glog/logging.h>
#include <gtest/gtest.h>

using namespace facebook::cuda;
using namespace std;
using namespace facebook::deeplearning::torch;
using namespace thpp;

namespace facebook { namespace deeplearning { namespace torch { namespace test {

template<int Dim>
std::pair<Tensor<float>, Tensor<float>>
makeRandomResized(initializer_list<long> size, initializer_list<long> resize) {
  CHECK_EQ(4, size.size()); // makeRandomTestTensor only implemented for 4 atm
  CHECK_EQ(Dim, resize.size()); // resize to proper size
  auto t = makeRandomTestTensor(size);
  t.resize(LongStorage(resize));
  auto tt = makeRandomTestTensor(size);
  tt.resize(LongStorage(resize));
  return make_pair(t, tt);
}

template<int Dim>
std::pair<std::unique_ptr<THCudaTensor, CudaTensorDeleter>,
          std::unique_ptr<THCudaTensor, CudaTensorDeleter>>
  transposeAndResize(Tensor<float> t,
                     initializer_list<long> resizeTransposed,
                     int sep,
                     Tensor<float>& tt,
                     bool asComplex = false) {
  CHECK_EQ(Dim, t.ndims());
  CHECK_EQ(Dim, tt.ndims());
  auto tCuda = copyToCuda(nullptr, t);
  auto ttCuda = copyToCuda(nullptr, tt);
  DeviceTensor<float, Dim> tCudaTensor =
    torchToDeviceTensor<float, Dim>(nullptr, tCuda.get());
  DeviceTensor<float, Dim> ttCudaTensor =
    torchToDeviceTensor<float, Dim>(nullptr, ttCuda.get());

  transpose(tCudaTensor, ttCudaTensor, sep, asComplex);
  tt = copyFromCuda(nullptr, ttCuda.get());
  tt.resize(LongStorage(resizeTransposed));
  return make_pair(std::move(tCuda), std::move(ttCuda));
}

template<int Dim>
void unTransposeAndCheckOutOfPlace(
    std::pair<Tensor<float>, Tensor<float>> pTensor,
    const std::pair<std::unique_ptr<THCudaTensor, CudaTensorDeleter>,
            std::unique_ptr<THCudaTensor, CudaTensorDeleter>>& pCudaTensor,
    int sep,
    initializer_list<long> testSize,
    bool asComplex = false) {
  auto ct = torchToDeviceTensor<float, Dim>(nullptr, pCudaTensor.first.get());
  auto ctt = torchToDeviceTensor<float, Dim>(nullptr, pCudaTensor.second.get());

  transpose(ct, ctt, Dim - sep, asComplex);
  pTensor.second = copyFromCuda(nullptr, pCudaTensor.first.get());
  pTensor.first.resize(LongStorage(testSize));
  pTensor.second.resize(LongStorage(testSize));

  auto comparison = compareTensors(pTensor.first, pTensor.second, 0);
  ASSERT_TRUE(comparison.first) << "Mismatch " << pTensor.first <<
    pTensor.second << comparison.second;
}

TEST(TransposeTest, test2D) {
  long s1 = 3;
  long s2 = 4;
  const int kDim = 2;
  const int kDimTranspose = 1;
  // Size for alloc, must be 4-D
  auto size = {(long)1, (long)1, s1, s2};
  // Interpreted as 2-D
  auto resize = {s1, s2};
  // Transpose to 2-D
  auto resizeTransposed = {s2, s1};
  // Size for test, must be 4-D
  auto testSize = size;

  auto pTensor = makeRandomResized<kDim>(size, resize);
  auto pCudaTensor = transposeAndResize<kDim>(
    pTensor.first, resizeTransposed, kDimTranspose,
    pTensor.second);

  // Specific transpose check
  for (auto i = 0; i < pTensor.first.size(0); ++i) {
    for (auto j = 0; j < pTensor.first.size(1); ++j) {
      ASSERT_EQ(pTensor.first.at({i, j}), pTensor.second.at({j, i})) <<
        " i = " << i << " j = " << j;
    }
  }

  // Out-of place transpose back and check
  unTransposeAndCheckOutOfPlace<kDim>(
    pTensor, pCudaTensor, kDimTranspose, testSize);
}

TEST(TransposeTest, test3D) {
  long s1 = 3;
  long s2 = 4;
  long s3 = 6;
  const int kDim = 3;
  const int kDimTranspose = 1;
  // Size for alloc, must be 4-D
  auto size = {(long)1, s1, s2, s3};
  auto resize = {s1, s2, s3};
  auto resizeTransposed = {s2, s3, s1};
  // Size for test, must be 4-D
  auto testSize = size;

  auto pTensor = makeRandomResized<kDim>(size, resize);
  auto pCudaTensor = transposeAndResize<kDim>(
    pTensor.first, resizeTransposed, kDimTranspose,
    pTensor.second);

  // Specific transpose check
  for (auto i = 0; i < pTensor.first.size(0); ++i) {
    for (auto j = 0; j < pTensor.first.size(1); ++j) {
      for (auto k = 0; k < pTensor.first.size(2); ++k) {
        ASSERT_EQ(pTensor.first.at({i, j, k}),
                  pTensor.second.at({j, k, i})) <<
          " i = " << i << " j = " << j << " k = " << k;
      }
    }
  }

  // Out-of place transpose back and check
  unTransposeAndCheckOutOfPlace<kDim>(
    pTensor, pCudaTensor, kDimTranspose, testSize);
}

TEST(TransposeTest, test5D) {
  long s1 = 3;
  long s2 = 4;
  long s3 = 5;
  long s4 = 6;
  long s5 = 3;
  const int kDim = 5;
  const int kDimTranspose = 3;
  // Size for alloc, must be 4-D
  auto size = {s1, s2, s3, s4 * s5};
  // Interpreted as 5-D
  auto resize = {s1, s2, s3, s4, s5};
  // Transpose to 5-D
  auto resizeTransposed = {s4, s5, s1, s2, s3};
  // Size for test, must be 4-D
  auto testSize = size;

  auto pTensor = makeRandomResized<kDim>(size, resize);
  auto pCudaTensor = transposeAndResize<kDim>(
    pTensor.first, resizeTransposed, kDimTranspose,
    pTensor.second);

  for (auto i = 0; i < pTensor.first.size(0); ++i) {
    for (auto j = 0; j < pTensor.first.size(1); ++j) {
      for (auto k = 0; k < pTensor.first.size(2); ++k) {
        for (auto l = 0; l < pTensor.first.size(3); ++l) {
          for (auto m = 0; m < pTensor.first.size(4); ++m) {
            CHECK_EQ(pTensor.first.at({i, j, k, l, m}),
                     pTensor.second.at({l, m, i, j, k})) <<
              " i = " << i << " j = " << j << " k = " << k <<
              " l = " << l << " m = " << m;
          }
        }
      }
    }
  }

  // Out-of place transpose back and check
  unTransposeAndCheckOutOfPlace<kDim>(
    pTensor, pCudaTensor, kDimTranspose, testSize);
}

TEST(TransposeTest, testComplex3D) {
  long s1 = 3;
  long s2 = 4;
  long s3 = 5;
  long s4 = 6;

  const int kDim = 3;
  const int kDimTranspose = 1;
  const long kComplexSize = 2;
  // Size for alloc, must be 4-D
  auto size = {s1, s2, s3, s4};
  // Interpreted as 5-D
  auto resize = {s1 * s2 * s3, s4 / kComplexSize, kComplexSize};
  // Transpose to 5-D
  auto resizeTransposed = {s4 / kComplexSize, s1 * s2 * s3, kComplexSize};
  // Size for test, must be 4-D
  auto testSize = size;

  auto pTensor = makeRandomResized<kDim>(size, resize);
  auto pCudaTensor = transposeAndResize<kDim>(
    pTensor.first, resizeTransposed, kDimTranspose,
    pTensor.second, true);

  for (auto i = 0; i < pTensor.first.size(0); ++i) {
    for (auto j = 0; j < pTensor.first.size(1); ++j) {
      // loadAs in thpp ?
      cuFloatComplex tf = *(cuFloatComplex*)(&pTensor.first.at({i, j, 0}));
      cuFloatComplex ttf = *(cuFloatComplex*)(&pTensor.second.at({j, i, 0}));
      ASSERT_EQ(tf.x, ttf.x) <<
        " i = " << i <<
        " j = " << j;
      ASSERT_EQ(tf.y, ttf.y) <<
        " i = " << i <<
        " j = " << j;
    }
  }

  // Out-of place transpose back and check
  unTransposeAndCheckOutOfPlace<kDim>(
    pTensor, pCudaTensor, kDimTranspose, testSize, true);
}

TEST(TransposeTest, testComplex5D) {
  long s1 = 1;
  long s2 = 3;
  long s3 = 9;
  long s4 = 9;
  long s5 = 2;

  const int kDim = 5;
  const int kDimTranspose = 2;
  const long kComplexSize = 2;
  // Size for alloc, must be 4-D
  auto size = {s1, s2, s3, s4 * s5};
  // Interpreted as 5-D
  auto resize = {s1, s2, s3, (s4 * s5) / kComplexSize, kComplexSize};
  // Transpose to 5-D
  auto resizeTransposed = {s3, (s4 * s5) / kComplexSize, s1, s2, kComplexSize};
  // Size for test, must be 4-D
  auto testSize = size;

  auto pTensor = makeRandomResized<kDim>(size, resize);
  auto pCudaTensor = transposeAndResize<kDim>(
    pTensor.first, resizeTransposed, kDimTranspose,
    pTensor.second, true);

  for (auto i = 0; i < pTensor.first.size(0); ++i) {
    for (auto j = 0; j < pTensor.first.size(1); ++j) {
      for (auto k = 0; k < pTensor.first.size(2); ++k) {
        for (auto l = 0; l < pTensor.first.size(3); ++l) {
          // loadAs in thpp ?
          cuFloatComplex tf =
            *(cuFloatComplex*)(&pTensor.first.at({i, j, k, l, 0}));
          cuFloatComplex ttf =
            *(cuFloatComplex*)(&pTensor.second.at({k, l, i, j, 0}));
          ASSERT_EQ(tf.x, ttf.x) << " i = " << i << " j = " << j <<
            " k = " << k << " l = " << l;
          ASSERT_EQ(tf.x, ttf.x) << " i = " << i << " j = " << j <<
            " k = " << k << " l = " << l;
        }
      }
    }
  }

  // Out-of place transpose back and check
  unTransposeAndCheckOutOfPlace<kDim>(
    pTensor, pCudaTensor, kDimTranspose, testSize, true);
}

} } } } // namespace

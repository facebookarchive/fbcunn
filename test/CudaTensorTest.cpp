// Copyright 2004-present Facebook. All Rights Reserved.
#include "torch/fb/fbcunn/src/CudaTensorUtils.h"
#include "THC.h"
#include "torch/fb/fbcunn/test/CudaTensorTestKernels.cuh"
#include "folly/Optional.h"
#include "folly/ScopeGuard.h"

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <vector>

using namespace std;

namespace facebook { namespace deeplearning { namespace torch {

namespace {

void verify3d(THCudaTensor* tensor) {
  auto hostStorage = THFloatStorage_newWithSize(tensor->storage->size);
  SCOPE_EXIT{ THFloatStorage_free(hostStorage); };

  THFloatStorage_copyCuda(nullptr, hostStorage, tensor->storage);

  for (int k = 0; k < tensor->size[0]; ++k) {
    for (int j = 0; j < tensor->size[1]; ++j) {
      for (int i = 0; i < tensor->size[2]; ++i) {
        // Values per entry are unique because the dimensions are
        // different and prime
        EXPECT_EQ(
          k * tensor->size[0] + j * tensor->size[1] + i * tensor->size[2],
          hostStorage->data[k * tensor->stride[0] +
                            j * tensor->stride[1] +
                            i * tensor->stride[2]]);
      }
    }
  }
}

} // unnamed namespace

TEST(CudaTensor, testDimensionMismatch) {
  EXPECT_THROW(testAssignment3d(nullptr,
                 makeTHCudaTensorFull(nullptr, {1, 2, 3, 4}).get()),
               invalid_argument);
  EXPECT_THROW(testAssignment3d(nullptr,
                 makeTHCudaTensorFull(nullptr, {1}).get()),
               invalid_argument);
}

TEST(CudaTensor, testWrite3d) {
  auto tensor = makeTHCudaTensorFull(nullptr, {11, 7, 5});

  // Run our kernel
  EXPECT_TRUE(testAssignment3d(nullptr, tensor.get()));
  verify3d(tensor.get());
}

TEST(CudaTensor, testWrite3dNonTrivialStride) {
  auto tensor = makeTHCudaTensorFull(nullptr, {11, 7, 5}, {200, 6, 1});

  // Run our kernel
  EXPECT_TRUE(testAssignment3d(nullptr, tensor.get()));
  verify3d(tensor.get());
}

TEST(CudaTensor, testWrite1d) {
  constexpr long kSize = 3;
  auto storage = THCudaStorage_newWithSize(nullptr, kSize);
  auto tensor = THCudaTensor_newWithStorage1d(nullptr, storage, 0, kSize, 1);
  SCOPE_EXIT{ THCudaTensor_free(nullptr, tensor); };

  // Clear out tensor
  THCudaTensor_fill(nullptr, tensor, 0.0f);

  // Run our kernel
  EXPECT_TRUE(testAssignment1d(nullptr, tensor));

  // Verify output
  auto hostStorage = THFloatStorage_newWithSize(tensor->storage->size);
  SCOPE_EXIT{ THFloatStorage_free(hostStorage); };

  THFloatStorage_copyCuda(nullptr, hostStorage, storage);

  for (int i = 0; i < tensor->size[0]; ++i) {
    EXPECT_EQ(i, hostStorage->data[i]);
  }
}

TEST(CudaTensor, testUpcast) {
  // test with no padding
  EXPECT_TRUE(testUpcast(nullptr,
                makeTHCudaTensorFull(nullptr, {3, 2, 1}).get()));

  // test with padding
  EXPECT_TRUE(testUpcast(nullptr,
                makeTHCudaTensorFull(nullptr, {4, 3, 2}, {150, 40, 15}).get()));
}

TEST(CudaTensor, testDowncastIllegalPaddingThrows) {
  // 16 should be 12 for no padding
  EXPECT_THROW(testDowncastTo2d(nullptr,
                 makeTHCudaTensorFull(nullptr, {2, 3, 4}, {16, 4, 1}).get()),
               invalid_argument);

  // 15/5 should be 12/3 for no padding
  EXPECT_THROW(testDowncastTo1d(nullptr,
                 makeTHCudaTensorFull(nullptr, {2, 3, 4}, {15, 5, 1}).get()),
               invalid_argument);

  // But, the same should not cause a problem for 2d since the padding
  // is in the non-collapsed dimensions
  EXPECT_NO_THROW(testDowncastTo2d(nullptr,
                    makeTHCudaTensorFull(
                      nullptr, {2, 3, 4}, {15, 5, 1}).get()));
}

TEST(CudaTensor, testDowncast) {
  EXPECT_TRUE(testDowncastTo2d(nullptr,
                               makeTHCudaTensorFull(nullptr, {2, 3, 4}).get()));

  // We can have padding in the innermost dimension
  EXPECT_TRUE(testDowncastTo2d(nullptr,
                               makeTHCudaTensorFull(nullptr, {2, 3, 4},
                                                    {36, 12, 3}).get()));
}

TEST(CudaTensor, testDowncastWrites) {
  auto tensor = makeTHCudaTensorFull(nullptr, {2, 3, 4});
  EXPECT_TRUE(testDowncastWrites(nullptr, tensor.get()));

  // Verify output
  auto hostStorage = THFloatStorage_newWithSize(tensor->storage->size);
  SCOPE_EXIT{ THFloatStorage_free(hostStorage); };

  THFloatStorage_copyCuda(nullptr, hostStorage, tensor->storage);

  // In the downcast view, we should have overwritten all the values
  for (int k = 0; k < tensor->size[0]; ++k) {
    for (int j = 0; j < tensor->size[1]; ++j) {
      for (int i = 0; i < tensor->size[2]; ++i) {
        EXPECT_EQ(1.0f,
                  hostStorage->data[k * tensor->stride[0] +
                                    j * tensor->stride[1] +
                                    i * tensor->stride[2]]);
      }
    }
  }
}

} } } // namespace

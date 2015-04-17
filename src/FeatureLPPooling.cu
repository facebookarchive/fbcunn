// Copyright 2004-present Facebook. All Rights Reserved.

#include "FeatureLPPooling.cuh"
#include "cuda/DeviceTensor.cuh"
#include "cuda/CudaStaticAssert.cuh"
#include "cuda/CudaUtils.cuh"
#include "cuda/RegisterUtils.cuh"
#include "util/Misc.h"
#include "THC.h"

#include <boost/preprocessor/repetition/repeat.hpp>

using namespace facebook::cuda;

#define OUTPUT_FEATURES_PER_THREAD 32
#define MAX_WARPS_PER_RUN 4

namespace facebook { namespace deeplearning { namespace torch {

namespace detail {

__device__ __forceinline__
int getDim1Point(const DeviceTensor<float, 4>& input) {
  const int threadPoint = blockIdx.x * blockDim.x + threadIdx.x;
  return threadPoint / input.getSize(3);
}

__device__ __forceinline__
int getDim2Point(const DeviceTensor<float, 4>& input) {
  const int threadPoint = blockIdx.x * blockDim.x + threadIdx.x;
  return threadPoint % input.getSize(3);
}

__device__ __forceinline__
int getStartOutputFeature() {
  return blockIdx.y * OUTPUT_FEATURES_PER_THREAD;
}

__device__ __forceinline__
int getEndOutputFeature(const DeviceTensor<float, 4>& output) {
  return min((blockIdx.y + 1) * OUTPUT_FEATURES_PER_THREAD, output.getSize(1));
}

__device__ __forceinline__
int getBatch() {
  return blockIdx.z;
}

// All of these functions that follow are MathOps; they are template
// parameters so L2 can be more efficiently implemented
typedef float (*MathOp)(const float in, const float arg);

__device__ __forceinline__ float power2(const float in, const float power) {
  return in * in;
}

__device__ __forceinline__ float root2(const float in, const float power) {
  return sqrtf(in);
}

__device__ __forceinline__ float powerGrad2(const float in, const float power) {
  return in;
}

__device__ __forceinline__ float powerN(const float in, const float power) {
  return powf(in, power);
}

__device__ __forceinline__ float rootN(const float in, const float power) {
  const float invPower = 1.0f / power;
  return powf(in, invPower);
}

__device__ __forceinline__ float powerGradN(const float in, const float power) {
  return powf(in, power - 1.0f);
}

// Input is of the form:
// [batch][feature dim][optional dim 1][optional dim 2]
template <int Width, int Stride, MathOp PowerFunc, MathOp RootFunc>
__global__ void
featureLPPoolingUpdateOutput(const DeviceTensor<float, 4> input,
                             DeviceTensor<float, 4> output,
                             float power) {
  // What non-feature points is this thread handling?
  const int dim1Point = getDim1Point(input);
  const int dim2Point = getDim2Point(input);

  if (dim1Point >= input.getSize(2) || dim2Point >= input.getSize(3)) {
    // This thread in the warp is out of bounds
    return;
  }

  // What feature points is this thread handling?
  const int startOutputFeature = getStartOutputFeature();
  const int endOutputFeature = getEndOutputFeature(output);
  const int startInputFeature = startOutputFeature * Stride;

  // What batch points is this thread handling?
  const int batch = getBatch();

  // If stride >= width, then there is no loaded data reuse.
  // If stride > 1 and stride < width, then shift by stride, since we
  // can reuse Width - Stride elements from the previous round.
  // e.g., width = 5, stride = 2,
  // output 0 uses input 0 1 2 3 4
  // output 1 uses input 2 3 4 5 6 (inputs 2 - 4 are reused, i.e., 5 -
  // 2 elements are reused, and we have to shift the array by 2)
  //
  // e.g., width = 5, stride = 3,
  // output 0 uses input 0 1 2 3 4
  // output 1 uses input 3 4 5 6 7 (inputs 3 - 4 are reused, i.e., 5 - 3
  // elements are reused, and we have to shift the array by 3)

  // Valid only pooling: load Width elements from input (Width -
  // Stride is handled here, at the top of the loop we handle the
  // remaining Stride elements). We already verified that the input is
  // larger than the width.
  // `in` will contain the input values ^ power.
  float in[Width];

#pragma unroll
  for (int i = 0; i < Width - Stride; ++i) {
    const float data =
      input[batch][startInputFeature + i][dim1Point][dim2Point];
    in[i] = PowerFunc(data, power);
  }

  for (int outputFeature = startOutputFeature;
       outputFeature < endOutputFeature;
       ++outputFeature) {
    // If Stride < Width, we're loading Stride new values starting at
    // Width - Stride
    // If Stride >= Width, we're loading Width new values starting at 0
    if (Stride < Width) {
      const int nextInputFeature = outputFeature * Stride + Width - Stride;

#pragma unroll
      for (int i = 0; i < Stride; ++i) {
        const float data =
          input[batch][nextInputFeature + i][dim1Point][dim2Point];
        in[Width - Stride + i] = PowerFunc(data, power);
      }
    } else {
      const int nextInputFeature = outputFeature * Stride;

#pragma unroll
      for (int i = 0; i < Width; ++i) {
        float data = input[batch][nextInputFeature + i][dim1Point][dim2Point];
        in[i] = PowerFunc(data, power);
      }
    }

    // Calculate the new output feature
    float val = 0.0f;
    for (int i = 0; i < Width; ++i) {
      val += in[i];
    }

    val = RootFunc(val, power);
    output[batch][outputFeature][dim1Point][dim2Point] = val;

    if (Stride < Width) {
      // Shift registers for calculating the next point
      RegisterUtils<float, Width>::shiftLeft<Stride>(in);
    }
  }
}

// forward pass: f(a, ..., z) = (a^p + ... + z^p)^(1 / p)
// for bprop:
//   partial df(a, ... z)/da = a^(p - 1) * (a^p + ... + z^p)^((1 / p) - 1) =
//   a^(p - 1) * 1/(f(a, ..., z)^(p - 1)) = (a / f(a, ..., z))^(p - 1)
//
// example: for p = 2, df(a, ..., z)/da = a / f(a, ..., z)
// example: for p = 3, df(a, ..., z)/da = (a / f(a, ..., z))^2
//
// PowerGradFunc implements x^(p - 1)
template <int Width, int Stride, MathOp PowerGradFunc>
__launch_bounds__(32 * 8, 8) // max 32 registers per thread
__global__ void
featureLPPoolingUpdateGradInput(const DeviceTensor<float, 4> gradOutput,
                                const DeviceTensor<float, 4> input,
                                const DeviceTensor<float, 4> output,
                                DeviceTensor<float, 4> gradInput,
                                float power) {
  // What non-feature points is this thread handling?
  const int dim1Point = getDim1Point(input);
  const int dim2Point = getDim2Point(input);

  if (dim1Point >= input.getSize(2) || dim2Point >= input.getSize(3)) {
    // This thread in the warp is out of bounds
    return;
  }

  // What feature points is this thread handling? [start, end)
  const int startOutputFeature = getStartOutputFeature();
  const int endOutputFeature = getEndOutputFeature(output);

  // What is the first input point that the output features depend
  // upon? [start, end)
  const int startInputFeature = startOutputFeature * Stride;
  const int endInputFeature = endOutputFeature * Stride;

  // What batch points is this thread handling?
  const int batch = getBatch();

  // atomicAdd into gradInput is slow, avoid it where possible.
  // We can do this because there is a range of gradInput elements
  // that we are updating exclusively. This is how we find it
  //
  //  width = 3 stride = 1 example:
  // ------------------------------
  //      startOutputFeature for this thread
  //        |
  //        |
  // previous thread's output feature
  //   |    |
  //   |    |                  gradOutput
  // __v____v___________________
  // |    |    |    |    |    |
  // ---------------------------
  //   |\ \_____
  //   | \__    \               gradInput
  // __v____v____v_____________
  // |    |    |    |    |    |
  // ---------------------------
  //         A        A
  //         |        |
  //    startInputFeature
  //                  |
  //                  exclusiveStartInputFeature
  //
  // exclusiveStartInputFeature is the first input feature that we can
  // write into exclusively; the one right before it overlaps with
  // updates from a previous thread and thus has to use atomicAdd.
  const int exclusiveStartInputFeature =
    startInputFeature == 0 ?
    // no thread is before ourselves
    0 :
    // there is a thread before ourselves
    startInputFeature + (Width - 1) * Stride;

  // Similarly, exclusiveEndInputFeature is the last input feature
  // that we can write into exclusively, since we might be overlapping
  // with the following thread
  const int exclusiveEndInputFeature =
    endOutputFeature == output.getSize(1) ?
    // no thread is after ourselves
    endInputFeature + (Width - 1) * Stride :
    // there is a thread after ourselves
    endInputFeature;

  // As with updateOutput preload input elements, except no need to
  // transform them
  float in[Width];
#pragma unroll
  for (int i = 0; i < Width - Stride; ++i) {
    in[i] = input[batch][startInputFeature + i][dim1Point][dim2Point];
  }

  for (int outputFeature = startOutputFeature;
       outputFeature < endOutputFeature;
       ++outputFeature) {
    // As with updateOutput load the subsequent input elements that we
    // need, except no need to transform them
    //
    // If Stride < Width, we're loading Stride new values starting at
    // Width - Stride
    // If Stride >= Width, we're loading Width new values starting at 0
    if (Stride < Width) {
      const int nextInputFeature = outputFeature * Stride + Width - Stride;

#pragma unroll
      for (int i = 0; i < Stride; ++i) {
        in[Width - Stride + i] =
          input[batch][nextInputFeature + i][dim1Point][dim2Point];
      }
    } else {
      const int nextInputFeature = outputFeature * Stride;

#pragma unroll
      for (int i = 0; i < Width; ++i) {
        in[i] = input[batch][nextInputFeature + i][dim1Point][dim2Point];
      }
    }

    // A given output feature gradient contributes to `Width` input
    // gradients
    const float gradOut =
      gradOutput[batch][outputFeature][dim1Point][dim2Point];

    // Load output (f(x_is)). It is possible that this is zero, in
    // which case we'll ignore this point.
    float out = output[batch][outputFeature][dim1Point][dim2Point];
    if (out == 0.0f) {
      continue;
    }

    const int curStartInputFeature = outputFeature * Stride;
    const int curEndInputFeature = outputFeature * Stride + Width - 1;

    if (curStartInputFeature >= exclusiveStartInputFeature &&
        curEndInputFeature < exclusiveEndInputFeature) {
      // This thread is exclusively responsible for updating these
      // input points, so we need not make the addition atomic
      for (int i = 0; i < Width; ++i) {
        const int inputFeature = outputFeature * Stride + i;

        // Calculate grad * (x_i / f(x_is))^(p - 1)
        const float val = gradOut * PowerGradFunc(in[i] / out, power);

        gradInput[batch][inputFeature][dim1Point][dim2Point] += val;
      }
    } else {
      // Handle start and end boundary cases: potential overlap with
      // other threads
      for (int i = 0; i < Width; ++i) {
        const int inputFeature = outputFeature * Stride + i;

        // Calculate grad * (x_i / f(x_is))^(p - 1)
        const float val = gradOut * PowerGradFunc(in[i] / out, power);

        // We don't overlap other threads for this range
        if (inputFeature >= exclusiveStartInputFeature &&
            inputFeature < exclusiveEndInputFeature) {
          gradInput[batch][inputFeature][dim1Point][dim2Point] += val;
        } else {
          // We are potentially overlapping with threads handling
          // features before ourselves, so these need to be added atomically
          atomicAdd(&gradInput[batch][inputFeature][dim1Point][dim2Point],
                    val);
        }
      }
    }

    if (Stride < Width) {
      // Shift registers for calculating the next point
      RegisterUtils<float, Width>::shiftLeft<Stride>(in);
    }
  }
}

} // namespace detail

bool
runFeatureLPPoolingUpdateOutput(cudaStream_t stream,
                                const DeviceTensor<float, 4>& input,
                                DeviceTensor<float, 4>& output,
                                float power, int width, int stride) {
  const cudaDeviceProp& deviceProperties =
    facebook::CUDAUtil::getCurrentDeviceProperties();
  const int outputFeatures = ((input.getSize(1) - width) / stride) + 1;

  assert(input.getSize(0) == output.getSize(0));
  assert(outputFeatures == output.getSize(1));
  assert(input.getSize(1) >= width);

  assert(input.getSize(2) == output.getSize(2));
  assert(input.getSize(3) == output.getSize(3));
  assert(power > 0.0f);
  assert(width >= 1);
  assert(stride >= 1);

  // Split non-features among threads and grid x
  int totalNonFeatureSize = input.getSize(2) * input.getSize(3);
  int numWarps =
    min(ceil(totalNonFeatureSize, deviceProperties.warpSize),
        MAX_WARPS_PER_RUN);
  int blockSize = deviceProperties.warpSize * numWarps;

  // Split non-features among grid x
  int nonFeatureSizeBlocks = ceil(totalNonFeatureSize, blockSize);

  // Split features among grid y, up to a maximum number of features per thread
  int featureBlocks = ceil(outputFeatures, OUTPUT_FEATURES_PER_THREAD);

  // Split batch among grid z.
  dim3 grid(nonFeatureSizeBlocks, featureBlocks, input.getSize(0));
  dim3 block(blockSize);

#define L2_STRIDE_CASE(UNUSED, STRIDE_MIN_1, WIDTH)                     \
  case STRIDE_MIN_1 + 1:                                                \
    detail::                                                            \
    featureLPPoolingUpdateOutput<WIDTH,                                 \
                                 STRIDE_MIN_1 + 1,                      \
                                 detail::power2,                        \
                                 detail::root2><<<grid, block, 0, stream>>>( \
                                   input, output, power);               \
    return true;

// WIDTH_MIN_2 is from 0 -> 14, but we want 2 -> 16
#define L2_WIDTH_CASE(UNUSED1, WIDTH_MIN_2, UNUSED2)            \
    case WIDTH_MIN_2 + 2:                                       \
      switch (stride) {                                         \
        BOOST_PP_REPEAT(4, L2_STRIDE_CASE, WIDTH_MIN_2 + 2);    \
      }

#define LP_STRIDE_CASE(UNUSED, STRIDE_MIN_1, WIDTH)                     \
  case STRIDE_MIN_1 + 1:                                                \
    detail::                                                            \
    featureLPPoolingUpdateOutput<WIDTH,                                 \
                                 STRIDE_MIN_1 + 1,                      \
                                 detail::powerN,                        \
                                 detail::rootN><<<grid, block, 0, stream>>>( \
                                   input, output, power);               \
    return true;

// WIDTH_MIN_2 is from 0 -> 14, but we want 2 -> 16
#define LP_WIDTH_CASE(UNUSED1, WIDTH_MIN_2, UNUSED2)            \
    case WIDTH_MIN_2 + 2:                                       \
      switch (stride) {                                         \
        BOOST_PP_REPEAT(4, LP_STRIDE_CASE, WIDTH_MIN_2 + 2);    \
      }

  if (power == 2.0f) {
    switch (width) {
      // widths 2 -> 16 (PP iterate from 0 -> 14)
      BOOST_PP_REPEAT(15, L2_WIDTH_CASE, 0);
    }
  } else {
    switch (width) {
      // widths 2 -> 16 (PP iterate from 0 -> 14)
      BOOST_PP_REPEAT(15, LP_WIDTH_CASE, 0);
    }
  }

  // Otherwise, we have an unhandled width and/or stride.
  return false;

#undef L2_STRIDE_CASE
#undef L2_WIDTH_CASE
#undef LP_STRIDE_CASE
#undef LP_WIDTH_CASE
}

bool
runFeatureLPPoolingUpdateGradInput(cudaStream_t stream,
                                   const DeviceTensor<float, 4>& gradOutput,
                                   const DeviceTensor<float, 4>& input,
                                   const DeviceTensor<float, 4>& output,
                                   DeviceTensor<float, 4>& gradInput,
                                   float power, int width, int stride) {
  const cudaDeviceProp& deviceProperties =
    facebook::CUDAUtil::getCurrentDeviceProperties();

  for (int i = 0; i < 4; ++i) {
    assert(gradOutput.getSize(i) == output.getSize(i));
    assert(gradInput.getSize(i) == input.getSize(i));
  }

  int outputFeatures = ((input.getSize(1) - width) / stride) + 1;

  assert(gradInput.getSize(0) == gradOutput.getSize(0));
  assert(outputFeatures == gradOutput.getSize(1));
  assert(gradInput.getSize(1) >= width);

  assert(gradInput.getSize(2) == gradOutput.getSize(2));
  assert(gradInput.getSize(3) == gradOutput.getSize(3));
  assert(power > 0.0f);
  assert(width >= 1);
  assert(stride >= 1);

  // Different threads are potentially adding into overlapping input
  // points, so we must clear out gradInput before continuing.
  gradInput.fillAsync(0.0f, stream);

  // Split non-features among threads and grid x
  int totalNonFeatureSize = input.getSize(2) * input.getSize(3);
  int numWarps =
    min(ceil(totalNonFeatureSize, deviceProperties.warpSize),
        MAX_WARPS_PER_RUN);
  int blockSize = deviceProperties.warpSize * numWarps;

  // Split non-features among grid x
  int nonFeatureSizeBlocks = ceil(totalNonFeatureSize, blockSize);

  // Split features among grid y, up to a maximum number of features per thread
  int featureBlocks = ceil(outputFeatures, OUTPUT_FEATURES_PER_THREAD);

  // Split batch among grid z.
  dim3 grid(nonFeatureSizeBlocks, featureBlocks, input.getSize(0));
  dim3 block(blockSize);

#define L2_STRIDE_CASE(UNUSED, STRIDE_MIN_1, WIDTH)                     \
  case STRIDE_MIN_1 + 1:                                                \
    detail::                                                            \
    featureLPPoolingUpdateGradInput<WIDTH,                              \
                                    STRIDE_MIN_1 + 1,                   \
                                    detail::powerGrad2><<<grid, block,  \
                                    0, stream>>>(                       \
                                      gradOutput, input, output,        \
                                      gradInput, power);                \
    return true;

// WIDTH_MIN_2 is from 0 -> 14, but we want 2 -> 16
#define L2_WIDTH_CASE(UNUSED1, WIDTH_MIN_2, UNUSED2)            \
    case WIDTH_MIN_2 + 2:                                       \
      switch (stride) {                                         \
        BOOST_PP_REPEAT(4, L2_STRIDE_CASE, WIDTH_MIN_2 + 2);    \
      }

#define LP_STRIDE_CASE(UNUSED, STRIDE_MIN_1, WIDTH)                     \
  case STRIDE_MIN_1 + 1:                                                \
    detail::                                                            \
    featureLPPoolingUpdateGradInput<WIDTH,                              \
                                    STRIDE_MIN_1 + 1,                   \
                                    detail::powerGradN><<<grid, block,  \
                                    0, stream>>>(                       \
                                      gradOutput, input, output,        \
                                      gradInput, power);                \
    return true;

// WIDTH_MIN_2 is from 0 -> 14, but we want 2 -> 16
#define LP_WIDTH_CASE(UNUSED1, WIDTH_MIN_2, UNUSED2)            \
    case WIDTH_MIN_2 + 2:                                       \
      switch (stride) {                                         \
        BOOST_PP_REPEAT(4, LP_STRIDE_CASE, WIDTH_MIN_2 + 2);    \
      }

  if (power == 2.0f) {
    switch (width) {
      // widths 2 -> 16 (PP iterate from 0 -> 14)
      BOOST_PP_REPEAT(15, L2_WIDTH_CASE, 0);
    }
  } else {
    switch (width) {
      // widths 2 -> 16 (PP iterate from 0 -> 14)
      BOOST_PP_REPEAT(15, LP_WIDTH_CASE, 0);
    }
  }

  // Otherwise, we have an unhandled width and/or stride.
  return false;

#undef L2_STRIDE_CASE
#undef L2_WIDTH_CASE
#undef LP_STRIDE_CASE
#undef LP_WIDTH_CASE
}

} } }

/**
 * Copyright 2014 Facebook
 */
#pragma once

#include "cuda/DeviceTensor.cuh"
#include "DeviceTensorUtils.h"
#include <cassert>

namespace facebook { namespace deeplearning { namespace torch {
namespace detail {

struct LocallyConnectedParam {
  int batchSize;
  int outputPlanes;
  int outputHeight;
  int outputWidth;
  int inputPlanes;
  int inputHeight;
  int inputWidth;
  int kernelHeight;
  int kernelWidth;
  int dH;
  int dW;
};

// Constants for DeviceTensor getSize(...) for input and output
// DeviceTensors
const int kBatchDim  = 0;
const int kHeightDim = 1;
const int kWidthDim  = 2;
const int kPlaneDim  = 3;

// Constants for weight DeviceTensor
const int kKernelOutputHeightDim = 0;
const int kKernelOutputWidthDim  = 1;
const int kKernelHeightDim       = 2;
const int kKernelWidthDim        = 3;
const int kKernelOutputPlaneDim  = 4;
const int kKernelPlaneDim        = 5;


// check if a pointer is aligned on an address divisible by alignement.
inline bool isAligned(const void* pointer, int alignment) {
  return reinterpret_cast<size_t>(pointer) % alignment == 0;
}

// Convert image batch tensors of float to float2/4.
//   The elements of the tensor are converted to type T (float2 or
// float4) and the size of the last dimension is reduced accordingly.
//
template <typename T>
facebook::cuda::DeviceTensor<T, 4>
convertImageBatch(facebook::cuda::DeviceTensor<float, 4>& t) {
  int dim = sizeof(T) / sizeof(float);
  assert(dim >= 0);
  assert(t.getSize(3) >= dim);
  int imageBatchSize[4] = {t.getSize(0), t.getSize(1),
                           t.getSize(2), t.getSize(3) / dim};
  return facebook::cuda::DeviceTensor<T, 4>(reinterpret_cast<T*>(t.data()),
                                            imageBatchSize);
}

// Convert weight tensor of float to float2/4.
//   The elements of the tensor are converted to type T (float2 or
// float4) and the size of the last dimension is reduced accordingly.
//
template <typename T>
facebook::cuda::DeviceTensor<T, 6>
convertWeight(facebook::cuda::DeviceTensor<float, 6>& w) {
  int dim = sizeof(T) / sizeof(float);
  assert(dim >= 0);
  assert(w.getSize(5) >= dim);
  int weightSize[6] = {w.getSize(0), w.getSize(1), w.getSize(2),
                       w.getSize(3), w.getSize(4), w.getSize(5) / dim};
  return facebook::cuda::DeviceTensor<T, 6>(reinterpret_cast<T*>(w.data()),
                                            weightSize);
}

// Is a given number of power of two (n = 2^k | k in N).
//
inline
int isPowerOfTwo(int n) {
  int pot = 1;
  while (2 * pot <= n) pot *= 2;
  return pot == n;
}

// Compute the greatest power-of-two less than or equal to n.
//
inline
int greatestPowerOfTwoLessEq(int n) {
  int pot = 1;
  while (2 * pot <= n) pot *= 2;
  return pot;
}

// -----------------------------------------------------------------------------
// Vector operations
//
// These are helper methods that allow formulating the various kernels
// operating on float, float2, and float4 templated in those types.
// -----------------------------------------------------------------------------

// Initialize a float to zero.
__device__ __host__
inline
void zero(float& x) {
  x = 0.0f;
}

// Initialize a float2 with zeros.
__device__ __host__
inline
void zero(float2& v) {
  v.x = 0.0f;
  v.y = 0.0f;
}

// Initialize a float4 with zeros.
__device__ __host__
inline
void zero(float4& v) {
  v.x = 0.0f;
  v.y = 0.0f;
  v.z = 0.0f;
  v.w = 0.0f;
}

__device__
inline
float2 operator* (float s, const float2& v) {
  float2 result = {s * v.x, s * v.y};
  return result;
}

__device__
inline
float4 operator* (float s, const float4& v) {
  float4 result = {s * v.x, s * v.y, s * v.z, s * v.w};
  return result;
}

__device__
inline
float2 operator +=(float2& s, const float2& v) {
  s.x += v.x;
  s.y += v.y;
  return s;
}

__device__
inline
float4 operator +=(float4& s, const float4& v) {
  s.x += v.x;
  s.y += v.y;
  s.z += v.z;
  s.w += v.w;
  return s;
}

__device__
inline
float dot(float a, float b) {
  return a * b;
}

__device__
inline
float dot(const float2& a, const float2& b) {
  return a.x * b.x + a.y * b.y;
}

__device__
inline
float dot(const float4& a, const float4& b) {
  return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}


// These constants are useful for running the full test suite against
// the less optimized code paths (by setting the optimization flags
// to false). As a consequence, for release code, one would expect all
// flags to be true, i.e. full optimizations enabled.
//
const bool kFloat4Optimization = true;
const bool kFloat2Optimization = true;

void locallyConnectedUpdateOutput(cudaStream_t stream,
                                  const float* input, const float* weight,
                                  const float* bias, float* output,
                                  LocallyConnectedParam& params);

void locallyConnectedUpdateGradInput(cudaStream_t stream,
                                     const float* gradOutput,
                                     const float* weight,
                                     float* gradInput,
                                     LocallyConnectedParam& params);

void locallyConnectedAccGradParameters(cudaStream_t stream,
                                       const float* input,
                                       const float* gradOutput,
                                       float* gradWeight,
                                       float* gradBias,
                                       float scale,
                                       LocallyConnectedParam& params);

} // detail namespace
}}}  // namespaces

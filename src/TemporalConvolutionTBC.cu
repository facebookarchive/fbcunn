// Copyright 2004-present Facebook. All Rights Reserved.
// Author: Benjamin Graham <benjamingraham@fb.com>

// Tensor formats
// Input: ilen * batchSize * inputPlanes
// Output: olen * batchSize * outputPlanes
// Weight: kw * inputPlanes * outputPlanes

#include "THC.h"
#include "cuda/CudaUtils.cuh"
//#include "cuda/WarpReductions.cuh"
//#include "cuda/util/CachedDeviceProperties.h"

#include "TemporalConvolutionTBC.cuh"

namespace facebook {
namespace deeplearning {
namespace torch {
namespace detail {

// kernels for forwarding and backwarding bias
__global__ void TemporalConvolutionTBC_fp_bias(
    float* output_features,
    float* bias,
    int output_stride,
    int rows) {
  int x = blockIdx.x * 32 + threadIdx.x;
  float b = bias[x];
  for (int row = blockIdx.y; row < rows; row += gridDim.y) {
    output_features[row * output_stride + x] = b;
  }
}
__global__ void TemporalConvolutionTBC_bp_bias(
    float* matrix,
    float* target,
    int rows,
    int stride,
    float scale) {
  int i = blockIdx.x * 32 + threadIdx.x;
  float t = 0;
  for (int j = blockIdx.y; j < rows; j += gridDim.y)
    t += matrix[j * stride + i];
  atomicAdd(&target[i], t * scale);
}

void runTemporalConvolutionTBC_updateOutput(
    THCState* state,
    const cuda::DeviceTensor<float, 3>& input,
    const cuda::DeviceTensor<float, 3>& output,
    const cuda::DeviceTensor<float, 3>& weight,
    const cuda::DeviceTensor<float, 1>& bias) {

  auto W = weight.data();
  auto B = bias.data();
  auto I = input.data();
  auto O = output.data();

  auto ilen = input.getSize(0);
  auto batchSize = input.getSize(1);
  auto inputPlanes = input.getSize(2);
  auto outputPlanes = output.getSize(2);
  auto olen = output.getSize(0);
  auto kw = weight.getSize(0);
  int pad = (olen - ilen + kw - 1) / 2;

  // input * weights + bias -> output_features

  int op32n = outputPlanes / 32;
  int op32r = outputPlanes % 32;
  if (op32n) {
    TemporalConvolutionTBC_fp_bias<<<
        dim3(op32n, 32),
        32,
        0,
        THCState_getCurrentStream(state)>>>(
          O, B, output.getStride(1), batchSize * olen);
  }
  if (op32r) {
    TemporalConvolutionTBC_fp_bias<<<
        dim3(1, 32),
        op32r,
        0,
        THCState_getCurrentStream(state)>>>(
          O + op32n * 32, B + op32n * 32, output.getStride(1), batchSize * olen);
  }

  for (int k = 0; k < kw; k++) {
    int iShift = std::max(0, k - pad);
    int oShift = std::max(0, pad - k);
    int t = std::min(ilen + pad - k, olen) - oShift;
    // Note: gemm assumes column-major matrices
    // input    is l*m (row-major)
    // weight   is m*r (row-major)
    // output   is l*r (row-major)
    if (t > 0)
      THCudaBlas_Sgemm(
          state,
          'n',
          'n',
          outputPlanes, // r
          batchSize * t, // l
          inputPlanes, // m
          1, // alpha
          W + k * weight.getStride(0),
          outputPlanes, // r
          I + iShift * input.getStride(0),
          input.getStride(1), // >=m
          1, // beta
          O + oShift * output.getStride(0),
          output.getStride(1) // r
          );
  }
}

void runTemporalConvolutionTBC_updateGradInput(
    THCState* state,
    const cuda::DeviceTensor<float, 3>& dInput,
    const cuda::DeviceTensor<float, 3>& dOutput,
    const cuda::DeviceTensor<float, 3>& weight) {
  auto ilen = dInput.getSize(0);
  auto batchSize = dInput.getSize(1);
  auto inputPlanes = dInput.getSize(2);
  auto outputPlanes = dOutput.getSize(2);
  auto olen = dOutput.getSize(0);
  auto kw = weight.getSize(0);
  int pad = (olen - ilen + kw - 1) / 2;

  auto W = weight.data();
  auto dI = dInput.data();
  auto dO = dOutput.data();

  for (int k = 0; k < kw; k++) {
    int iShift = std::max(0, k - pad);
    int oShift = std::max(0, pad - k);
    int t = std::min(ilen + pad - k, olen) - oShift;
    // dOutput * T(weight) -> dInput
    // Note: gemm assumes column-major matrices
    // dOutput is l*m (row-major)
    // weight  is r*m (row-major)
    // dInput  is l*r (row-major)
    if (t > 0)
      THCudaBlas_Sgemm(
          state,
          't',
          'n',
          inputPlanes, // r
          batchSize * t, // l
          outputPlanes, // m
          1, // alpha
          W + k * weight.getStride(0),
          outputPlanes, // m
          dO + oShift * dOutput.getStride(0),
          dOutput.getStride(1), // m
          1, // beta
          dI + iShift * dInput.getStride(0),
          dInput.getStride(1) // m
          );
  }
}

void runTemporalConvolutionTBC_accGradParameters(
    THCState* state,
    const cuda::DeviceTensor<float, 3>& input,
    const cuda::DeviceTensor<float, 3>& dOutput,
    const cuda::DeviceTensor<float, 3>& dWeight,
    const cuda::DeviceTensor<float, 1>& dBias,
    float scale) {
  auto ilen = input.getSize(0);
  auto batchSize = input.getSize(1);
  auto inputPlanes = input.getSize(2);
  auto outputPlanes = dOutput.getSize(2);
  auto olen = dOutput.getSize(0);
  auto kw = dWeight.getSize(0);
  int pad = (olen - ilen + kw - 1) / 2;

  auto dW = dWeight.data();
  auto dB = dBias.data();
  auto I = input.data();
  auto dO = dOutput.data();

  int op32n = outputPlanes / 32;
  int op32r = outputPlanes % 32;
  if (op32n) {
    TemporalConvolutionTBC_bp_bias<<<
        dim3(op32n, 32),
        32,
        0,
        THCState_getCurrentStream(state)>>>(
          dO, dB, batchSize * olen, dOutput.getStride(1), scale);
  }
  if (op32r) {
    TemporalConvolutionTBC_bp_bias<<<
        dim3(1, 32),
        op32r,
        0,
        THCState_getCurrentStream(state)>>>(
        dO + op32n * 32,
        dB + op32n * 32,
        batchSize * olen,
        dOutput.getStride(1),
        scale);
  }

  for (int k = 0; k < kw; k++) {
    int iShift = std::max(0, k - pad);
    int oShift = std::max(0, pad - k);
    int t = std::min(ilen + pad - k, olen) - oShift;
    // Note: gemm assumes column-major matrices
    // Input    is m*l (row-major)
    // dOutput  is m*r (row-major)
    // dWeight  is l*r (row-major)
    if (t > 0)
      THCudaBlas_Sgemm(
          state,
          'n',
          't',
          outputPlanes, // r
          inputPlanes, // l
          batchSize * t, // m
          scale, // alpha
          dO + oShift * dOutput.getStride(0),
          dOutput.getStride(1), // r
          I + iShift * input.getStride(0),
          input.getStride(1), // l
          1, // beta
          dW + k * dWeight.getStride(0),
          outputPlanes // r
          );
   }
}
}
}
}
} // namespaces

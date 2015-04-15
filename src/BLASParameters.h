// Copyright 2004-present Facebook. All Rights Reserved.
#pragma once

#include <cublas_v2.h>
#include <iostream>
#include <vector>

namespace facebook { namespace deeplearning { namespace torch {

// Column major: columns are contiguous in memory
// Cmxn <- Amxk * Bkxn becomes
// C'nxm <- A'kxm * B'nxk and so C'nxm <- B'nxk * A'kxm
struct BLASParameters {
  BLASParameters() :
      iterDims(0),
      batchDims(0),
      resourceIndex(0),
      batchStepA(1),
      batchStepB(1),
      batchStepC(1),
      scaleRe(1.0f),
      scaleIm(0.0f),
      asComplex(false),
      accumulate(false),
      handles(),
      streams(),
      transposeA(CUBLAS_OP_N),
      transposeB(CUBLAS_OP_N) {}

  // Outermost dimensions to be treated as individual iterations in enclosing
  // for loops.
  BLASParameters& withIterDims(int i) {
    iterDims = i;
    return *this;
  }
  // After iterDims, remaining outermost dimensions to be treated as batch
  // dimensions, for instance, in a gemmbatched call.
  BLASParameters& withBatchDims(int i) {
    batchDims = i;
    return *this;
  }
  // Force running on a particular handle / stream index in the handle /
  // stream vectors. The actual handle / stream we will end up running on is
  // recovered by modulo indexing into the vector, default handle / stream if
  // the vectors are empty.
  BLASParameters& withResourceIndex(int i) {
    resourceIndex = i;
    return *this;
  }
  // Distance between two batches of A, used in batched mode, in case we want
  // to compute one entry every k. Step of zerom means the same matrix A will
  // be read over and over again.
  BLASParameters& withBatchStepA(int i) {
    batchStepA = i;
    return *this;
  }
  // Distance between two batches of B, used in batched mode, in case we want
  // to compute one entry every k. Step of zerom means the same matrix B will
  // be read over and over again.
  BLASParameters& withBatchStepB(int i) {
    batchStepB = i;
    return *this;
  }
  // Distance between two batches of C, used in batched mode, in case we want
  // to compute one entry every k. Step of zerom means the same matrix C will
  // be written over and over again.
  BLASParameters& withBatchStepC(int i) {
    batchStepC = i;
    return *this;
  }
  // Sets real scale in C += alpha * C + scale * A * B
  BLASParameters& withScaleReal(float f) {
    scaleRe = f;
    return *this;
  }
  // Sets imaginary scale in C += alpha * C + scale * A * B
  BLASParameters& withScaleImaginary(float f) {
    scaleIm = f;
    return *this;
  }
  // Use cgemm instead of sgemm
  BLASParameters& withComplex(bool b) {
    asComplex = b;
    return *this;
  }
  // If true, computes C += scale * A * B. Default is C = scale * A * B.
  BLASParameters& withAccumulate(bool b) {
    accumulate = b;
    return *this;
  }
  // Set vector of handle resources
  BLASParameters& withHandles(const std::vector<cublasHandle_t>& h) {
    handles = h;
    return *this;
  }
  // Set vector of stream resources
  BLASParameters& withStreams(const std::vector<cudaStream_t>& s) {
    streams = s;
    return *this;
  }
  // Transpose A
  BLASParameters& withTransposeA(cublasOperation_t t) {
    transposeA = t;
    return *this;
  }
  // Transpose B
  BLASParameters& withTransposeB(cublasOperation_t t) {
    transposeB = t;
    return *this;
  }

  unsigned int iterDims;
  unsigned int batchDims;
  unsigned int resourceIndex;
  unsigned int batchStepA;
  unsigned int batchStepB;
  unsigned int batchStepC;
  float scaleRe;
  float scaleIm;
  bool asComplex;
  bool accumulate;
  std::vector<cublasHandle_t> handles;
  std::vector<cudaStream_t> streams;
  cublasOperation_t transposeA;
  cublasOperation_t transposeB;
};

std::ostream& operator<<(std::ostream& os, const BLASParameters& params);

}}}

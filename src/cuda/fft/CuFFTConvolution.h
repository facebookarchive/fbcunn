// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include "fft/CuFFTConvolution.cuh"

#include <algorithm>
#include <glog/logging.h>
#include <tuple>

namespace facebook { namespace deeplearning { namespace torch {

struct THParams {
  THParams(THCudaTensor* in,
           THCudaTensor* wei,
           THCudaTensor* out,
           THCudaTensor* b = nullptr,
           float sc = 0.0f);

  void free();

  THCudaTensor* input;
  THCudaTensor* weight;
  THCudaTensor* output;
  THCudaTensor* bias;
  float scale;
};

// The tensor which holds the result of the pass is often not allocated
// explicitly before entering this pass.
// Factor out this logic into this struct.
struct ProblemSizes {
  // Use this constructor to create whatever problem size you want and
  // allocate tensors
  ProblemSizes() :
      pass(ConvolutionPass::kUpdateOutput),
      batchSize(0),
      filterSize(0),
      planeSize(0),
      expandedSizeRow(0),
      expandedSizeCol(0),
      inputSizeRow(0),
      inputSizeCol(0),
      outputSizeRow(0),
      outputSizeCol(0),
      weightSizeRow(0),
      weightSizeCol(0)
    {}

  // Use this constructor if you already have input/output/filter/bias tensors
  explicit ProblemSizes(const THParams& params, ConvolutionPass p);

  ProblemSizes& withPass(ConvolutionPass p) {
    pass = p;
    return *this;
  }
  ProblemSizes& withBatch(long i) {
    batchSize = i;
    return *this;
  }
  ProblemSizes& withFilter(long i) {
    filterSize = i;
    return *this;
  }
  ProblemSizes& withPlane(long i) {
    planeSize = i;
    return *this;
  }
  ProblemSizes& withExpandedSizeRow(long i) {
    expandedSizeRow = i;
    return *this;
  }
  ProblemSizes& withExpandedSizeCol(long i) {
    expandedSizeCol = i;
    return *this;
  }
  ProblemSizes& withInputSizeRow(long i) {
    inputSizeRow = i;
    return *this;
  }
  ProblemSizes& withInputSizeCol(long i) {
    inputSizeCol = i;
    return *this;
  }
  ProblemSizes& withOutputSizeRow(long i) {
    outputSizeRow = i;
    return *this;
  }
  ProblemSizes& withOutputSizeCol(long i) {
    outputSizeCol = i;
    return *this;
  }
  ProblemSizes& withWeightSizeRow(long i) {
    weightSizeRow = i;
    return *this;
  }
  ProblemSizes& withWeightSizeCol(long i) {
    weightSizeCol = i;
    return *this;
  }

  long rows() const {
    return std::max({
        inputSizeRow, outputSizeRow, weightSizeRow, expandedSizeRow});
  }

  long cols() const {
    return std::max({
        inputSizeCol, outputSizeCol, weightSizeCol, expandedSizeCol});
  }

  std::vector<long> sizes() const;

  // This is used for autotuning, given a problem size fit
  THParams makeTensors() const;

  typedef unsigned int Pass;
  typedef std::tuple<long, long, long, long> InputSizesTuple;
  typedef std::tuple<long, long, long, long> FilterSizesTuple;
  typedef std::tuple<Pass, InputSizesTuple, FilterSizesTuple> ProblemSizesTuple;

  explicit operator ProblemSizesTuple() const {
    return std::make_tuple(
      pass.pass,
      std::make_tuple(batchSize, planeSize, inputSizeCol, inputSizeRow),
      std::make_tuple(filterSize, planeSize, weightSizeCol, weightSizeRow)
    );
  }

  ConvolutionPass pass;
  long batchSize;
  long filterSize; // a.k.a nOutputPlanes
  long planeSize;  // a.k.a nInputPlanes
  long expandedSizeRow; // frequency domain dimension
  long expandedSizeCol; // frequency domain dimension
  long inputSizeRow;
  long inputSizeCol;
  long outputSizeRow;
  long outputSizeCol;
  long weightSizeRow;
  long weightSizeCol;
};

struct CuFFTStrategy {
  CuFFTStrategy() : batch(true) {}

  explicit CuFFTStrategy(const ProblemSizes &p) :
      batch(true) {
    // Recover from a map or file if present or search and update the map /
    // file if not. By default just use the sizes passed to you.
    sizes = p;
  }

  static CuFFTStrategy& defaultStrategy() {
    static CuFFTStrategy defaultStrategy;
    return defaultStrategy;
  }

  CuFFTStrategy& withBatch(bool b) {
    batch = b;
    return *this;
  }

  std::vector<CuFFTStrategy> makeStrategies() const;

  THParams makeTensors() const {
    return sizes.makeTensors();
  }

  // Number of billions of reductions (virtual fmas) that would be computed in
  // the time domain.
  float GReductions() const {
    return ((float) (sizes.batchSize *
                     sizes.filterSize *
                     sizes.planeSize *
                     sizes.inputSizeCol *
                     sizes.inputSizeRow *
                     sizes.weightSizeRow *
                     sizes.weightSizeCol)) / 1e9;
  }

  long maxDataElements() const {
    return std::max({sizes.batchSize * sizes.filterSize,
          sizes.batchSize  * sizes.planeSize,
          sizes.filterSize * sizes.planeSize})
      * sizes.rows()
      * sizes.cols();
  }

  bool batch;
  ProblemSizes sizes;
};

std::ostream& operator<<(std::ostream& os, const ProblemSizes& p);
std::ostream& operator<<(std::ostream& os, const CuFFTStrategy& s);

}}}

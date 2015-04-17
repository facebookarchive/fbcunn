// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include "CuFFTConvolution.cuh"
#include "CuFFTWrapper.cuh"

#include <algorithm>
#include <glog/logging.h>
#include <tuple>

namespace facebook { namespace deeplearning { namespace torch {

struct THBuffers {
  THCudaTensor* input;
  THCudaTensor* inputTranspose;
  THCudaTensor* output;
  THCudaTensor* outputTranspose;
  THCudaTensor* weight;
  THCudaTensor* weightTranspose;
};

struct THParams {
  THParams(THCState* s,
           THCudaTensor* in,
           THCudaTensor* wei,
           THCudaTensor* out,
           THCudaTensor* b = nullptr,
           float sc = 0.0f,
           THBuffers buf = THBuffers());

  void free();

  THCState* state;
  THCudaTensor* input;
  THCudaTensor* weight;
  THCudaTensor* output;
  THCudaTensor* bias;
  float scale;
  THBuffers buffers;
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
  THParams makeTensors(THCState* state) const;

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
  THBuffers buffers;
};

struct CuFFTStrategy {
  static const auto constexpr kMaxElements = 80000000L;
  CuFFTStrategy() :
      mmVersion(CuFFTStrategy::MMVersion::fbTransposeMM),
      fftVersion(FFTParameters::FFTVersion::cufft)
    {}

  explicit CuFFTStrategy(const ProblemSizes &p) :
      mmVersion(CuFFTStrategy::MMVersion::fbTransposeMM),
      fftVersion(FFTParameters::FFTVersion::cufft)
    {
    // Recover from a map or file if present or search and update the map /
    // file if not. By default just use the sizes passed to you.
    sizes = p;
  }

  static CuFFTStrategy& defaultStrategy() {
    static CuFFTStrategy defaultStrategy;
    return defaultStrategy;
  }

  CuFFTStrategy& withBatchMM() {
    mmVersion = CuFFTStrategy::MMVersion::batchMM;
    return *this;
  }

  CuFFTStrategy& withManyMM() {
    mmVersion = CuFFTStrategy::MMVersion::manyMM;
    return *this;
  }

  CuFFTStrategy& withFBMM() {
    mmVersion = CuFFTStrategy::MMVersion::fbTransposeMM;
    return *this;
  }

  CuFFTStrategy& withFFT(FFTParameters::FFTVersion v) {
    fftVersion = v;
    return *this;
  }

  bool cufft() const { return fftVersion == FFTParameters::FFTVersion::cufft; }
  bool fbfft() const { return fftVersion == FFTParameters::FFTVersion::fbfft; }

  bool batchmm() const {
    return mmVersion == CuFFTStrategy::MMVersion::batchMM;
  }
  bool manymm()  const {
    return mmVersion == CuFFTStrategy::MMVersion::manyMM;
  }
  bool fbmm()    const {
    return mmVersion == CuFFTStrategy::MMVersion::fbTransposeMM;
  }

  std::vector<CuFFTStrategy> makeStrategies() const;

  THParams makeTensors(THCState* state) const {
    return sizes.makeTensors(state);
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

  enum MMVersion {
    batchMM = 0,
    manyMM = 1,
    fbTransposeMM = 2
  } mmVersion;
  FFTParameters::FFTVersion fftVersion;
  ProblemSizes sizes;

  CuFFTStrategy& withMM(CuFFTStrategy::MMVersion v) {
    mmVersion = v;
    return *this;
  }
};

std::ostream& operator<<(std::ostream& os, const ProblemSizes& p);
std::ostream& operator<<(std::ostream& os, const CuFFTStrategy& s);

}}}

// Copyright 2014 Facebook

#pragma once

#include "fft/CuFFTConvolution.h"

namespace facebook { namespace deeplearning { namespace torch {

struct SpatialConvolutionCuFFTTuner {
  static CuFFTStrategy getBestPerformance(ProblemSizes pbs);
};

}}}

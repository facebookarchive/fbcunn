// Copyright 2014 Facebook

#pragma once

#include "torch/fb/fbcunn/layers/cuda/fft/CuFFTConvolution.h"

namespace facebook { namespace deeplearning { namespace torch {
namespace detail {

void updateOutputTH(const THParams& p,
                    const ProblemSizes& originalSizes,
                    const CuFFTStrategy& s);

void updateGradInputTH(const THParams& p,
                       const ProblemSizes& originalSizes,
                       const CuFFTStrategy& s);

void accGradParametersTH(const THParams& p,
                         const ProblemSizes& originalSizes,
                         const CuFFTStrategy& s);

void cleanupBuffers();

}}}}

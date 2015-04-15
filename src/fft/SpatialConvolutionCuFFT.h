// Copyright 2014 Facebook

#pragma once

#include "torch/fb/fbcunn/src/fft/CuFFTStrategy.h"

namespace facebook { namespace deeplearning { namespace torch {
namespace detail {

void updateOutputTH(THCState* state,
                    const THParams& p,
                    const ProblemSizes& originalSizes,
                    const CuFFTStrategy& s);

void updateGradInputTH(THCState* state,
                       const THParams& p,
                       const ProblemSizes& originalSizes,
                       const CuFFTStrategy& s);

void accGradParametersTH(THCState* state,
                         const THParams& p,
                         const ProblemSizes& originalSizes,
                         const CuFFTStrategy& s);

void cleanupBuffers();

}}}}

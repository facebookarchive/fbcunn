// Copyright 2004-present Facebook. All Rights Reserved.
#pragma once

struct THCudaTensor;
struct THCState;

namespace facebook { namespace deeplearning { namespace torch {
namespace bias {

/// Applies an additive bias to all output elements, pointwise, one
/// bias per output plane
/// Performs the operation output[b][o][y][x] += bias[o]
void updateOutputBias(THCState* state,
                      THCudaTensor* outputTH,
                      THCudaTensor* biasTH);

/// Applies an additive bias to all output elements, pointwise, one
/// bias per kernel column.
/// Performs the operation output[b][o][x] += bias[x]
void updateOutputTemporalBias(THCState* state,
                              THCudaTensor* outputTH,
                              THCudaTensor* biasTH);

/// Updates the gradient bias with the scaled sum of the output per
/// output plane
/// Performs the operation gradBias[o] += biasScale * output[b][o][x][y]
void accGradParametersBias(THCState* state,
                           THCudaTensor* outputTH,
                           THCudaTensor* gradBiasTH,
                           float biasScale);

/// Updates the gradient bias with the scaled sum of the output per
/// output plane
/// Performs the operation gradBias[x] += biasScale * output[b][o][x]
void accGradParametersTemporalBias(THCState* state,
                                   THCudaTensor* outputTH,
                                   THCudaTensor* gradBiasTH,
                                   float biasScale);

} } } }

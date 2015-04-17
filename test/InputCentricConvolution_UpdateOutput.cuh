// Copyright 2004-present Facebook. All Rights Reserved.
#pragma once

struct THCudaTensor;
struct THCState;

namespace facebook { namespace deeplearning { namespace torch { namespace test {

bool InputCentricRelayoutConvolution_UpdateOutput(THCState* state,
                                                  THCudaTensor* inputTH,
                                                  THCudaTensor* kernelsTH,
                                                  long filterRowStride,
                                                  long filterColStride,
                                                  THCudaTensor* outputTH);

} } } }

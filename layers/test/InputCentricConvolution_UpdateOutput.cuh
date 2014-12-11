// Copyright 2004-present Facebook. All Rights Reserved.
#pragma once

struct THCudaTensor;

namespace facebook { namespace deeplearning { namespace torch { namespace test {

bool InputCentricRelayoutConvolution_UpdateOutput(THCudaTensor* inputTH,
                                                  THCudaTensor* kernelsTH,
                                                  long filterRowStride,
                                                  long filterColStride,
                                                  THCudaTensor* outputTH);

} } } }

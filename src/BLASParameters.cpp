// Copyright 2004-present Facebook. All Rights Reserved.

#include "BLASParameters.h"

using namespace std;

namespace facebook { namespace deeplearning { namespace torch {

std::ostream& operator<<(ostream& os, const BLASParameters& params) {
  os << " ITD = " << params.iterDims;
  os << " BTD = " << params.batchDims;
  os << " RIX = " << params.resourceIndex;
  os << " CPLX = " << params.asComplex;
  os << " batchStepA = " << params.batchStepA;
  os << " batchStepB = " << params.batchStepB;
  os << " batchStepC = " << params.batchStepC;
  os << " #handles = " << params.handles.size();
  os << " #streams = " << params.streams.size();
  os << " transposeA = " << ((params.transposeA == CUBLAS_OP_T) ? "t " :
                             (params.transposeA == CUBLAS_OP_C) ? "c " : "n");
  os << " transposeB = " << ((params.transposeB == CUBLAS_OP_T) ? "t " :
                             (params.transposeB == CUBLAS_OP_C) ? "c " : "n");
  os << " scale = (" << params.scaleRe << ", " << params.scaleIm << ")";
  return os;
}

}}}

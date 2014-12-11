/**
 * Copyright 2014 Facebook
 * @author Tudor Bosman (tudorb@fb.com)
 */

#ifndef DEEPLEARNING_TORCH_VML_H_
#define DEEPLEARNING_TORCH_VML_H_

#include <mkl.h>
#include "folly/Preprocessor.h"

namespace facebook { namespace deeplearning { namespace torch { namespace vml {

#define XO(name) name
#define XI(P, name) FB_CONCATENATE(P, name)

#define DEFINE_V(T, P, ours, theirs) \
  inline void XO(ours)(long n, const T* a, T* y) { \
    return XI(P, theirs)(n, a, y); \
  }

#define DEFINE_VV(T, P, ours, theirs) \
  inline void XO(ours)(long n, const T* a, const T* b, T* y) { \
    return XI(P, theirs)(n, a, b, y); \
  }

#define DEFINE_VS(T, P, ours, theirs) \
  inline void XO(ours)(long n, const T* a, T b, T* y) { \
    return XI(P,theirs)(n, a, b, y); \
  }

#define DEFINE_V2(T, P, ours, theirs) \
  inline void XO(ours)(long n, const T* a, T* y, T* z) { \
    return XI(P,theirs)(n, a, y, z); \
  }

#define DEFINE_OPS(T, P) \
  DEFINE_VV(T, P, add, Add) \
  DEFINE_VV(T, P, sub, Sub) \
  DEFINE_V(T, P, sqr, Sqr) \
  DEFINE_VV(T, P, mul, Mul) \
  DEFINE_V(T, P, abs, Abs) \
\
  DEFINE_V(T, P, inv, Inv) \
  DEFINE_VV(T, P, div, Div) \
  DEFINE_V(T, P, sqrt, Sqrt) \
  DEFINE_V(T, P, invSqrt, InvSqrt) \
  DEFINE_V(T, P, pow2o3, Pow2o3) \
  DEFINE_V(T, P, pow3o2, Pow3o2) \
  DEFINE_VV(T, P, pow, Pow) \
  DEFINE_VS(T, P, powx, Powx) \
  DEFINE_VV(T, P, hypot, Hypot) \
\
  DEFINE_V(T, P, exp, Exp) \
  DEFINE_V(T, P, expm1, Expm1) \
  DEFINE_V(T, P, ln, Ln) \
  DEFINE_V(T, P, log10, Log10) \
  DEFINE_V(T, P, log1p, Log1p) \
\
  DEFINE_V(T, P, sin, Sin) \
  DEFINE_V(T, P, cos, Cos) \
  DEFINE_V2(T, P, sinCos, SinCos) \
  DEFINE_V(T, P, tan, Tan) \
  DEFINE_V(T, P, asin, Asin) \
  DEFINE_V(T, P, acos, Acos) \
  DEFINE_V(T, P, atan, Atan) \
  DEFINE_VV(T, P, atan2, Atan2) \
\
  DEFINE_V(T, P, sinh, Sinh) \
  DEFINE_V(T, P, cosh, Cosh) \
  DEFINE_V(T, P, tanh, Tanh) \
  DEFINE_V(T, P, asinh, Asinh) \
  DEFINE_V(T, P, acosh, Acosh) \
  DEFINE_V(T, P, atanh, Atanh) \
\
  DEFINE_V(T, P, erf, Erf) \
  DEFINE_V(T, P, erfc, Erfc) \
  DEFINE_V(T, P, cdfNorm, CdfNorm) \
  DEFINE_V(T, P, erfInv, ErfInv) \
  DEFINE_V(T, P, erfcInv, ErfcInv) \
  DEFINE_V(T, P, cdfNormInv, CdfNormInv) \
  DEFINE_V(T, P, lGamma, LGamma) \
  DEFINE_V(T, P, tGamma, TGamma) \
\
  DEFINE_V(T, P, floor, Floor) \
  DEFINE_V(T, P, ceil, Ceil) \
  DEFINE_V(T, P, trunc, Trunc) \
  DEFINE_V(T, P, round, Round) \
  DEFINE_V(T, P, nearbyInt, NearbyInt) \
  DEFINE_V(T, P, rint, Rint) \
  DEFINE_V2(T, P, modf, Modf) \
  DEFINE_V(T, P, frac, Frac) \

DEFINE_OPS(float, vs)
DEFINE_OPS(double, vd)

#undef DEFINE_OPS
#undef DEFINE_V2
#undef DEFINE_VS
#undef DEFINE_VV
#undef DEFINE_V
#undef XI
#undef XO

}}}}  // namespaces

#endif /* DEEPLEARNING_TORCH_VML_H_ */

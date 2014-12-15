/**
 * Copyright 2014 Facebook
 * @author Tudor Bosman (tudorb@fb.com)
 */

#ifndef DEEPLEARNING_TORCH_BLAS_H_
#define DEEPLEARNING_TORCH_BLAS_H_

#include <mkl.h>
#include "folly/Preprocessor.h"

namespace facebook { namespace deeplearning { namespace torch { namespace blas {

#define XI(P, name) FB_CONCATENATE(FB_CONCATENATE(cblas_, P), name)
#define XI_I(P, name) FB_CONCATENATE(FB_CONCATENATE(cblas_i, P), name)

#define DEFINE_OPS(T, P) \
  inline T asum(long n, const T* x, long incx) { \
    return XI(P, asum)(n, x, incx); \
  } \
  inline void axpy(long n, T alpha, const T* x, long incx, T* y, long incy) { \
    XI(P, axpy)(n, alpha, x, incx, y, incy); \
  } \
  inline void copy(long n, const T* x, long incx, T* y, long incy) { \
    XI(P, copy)(n, x, incx, y, incy); \
  } \
  inline T dot(long n, const T* x, long incx, const T* y, long incy) { \
    return XI(P, dot)(n, x, incx, y, incy); \
  } \
  inline T nrm2(long n, const T* x, long incx) { \
    return XI(P, nrm2)(n, x, incx); \
  } \
  inline void rot(long n, T* x, long incx, T* y, long incy, T c, T s) { \
    XI(P, rot)(n, x, incx, y, incy, c, s); \
  } \
  inline void rotg(T* a, T* b, T* c, T* s) { \
    XI(P, rotg)(a, b, c, s); \
  } \
  inline void rotm(long n, T* x, long incx, T* y, long incy, const T* p) { \
    XI(P, rotm)(n, x, incx, y, incy, p); \
  } \
  inline void rotmg(T* d1, T* d2, T* x1, T y1, T* p) { \
    XI(P, rotmg)(d1, d2, x1, y1, p); \
  } \
  inline void scal(long n, T alpha, T* x, long incx) { \
    XI(P, scal)(n, alpha, x, incx); \
  } \
  inline void swap(long n, T* x, long incx, T* y, long incy) { \
    XI(P, swap)(n, x, incx, y, incy); \
  } \
  inline long iamax(long n, const T* x, long incx) { \
    return XI_I(P, amax)(n, x, incx); \
  } \
  inline long iamin(long n, const T* x, long incx) { \
    return XI_I(P, amin)(n, x, incx); \
  } \
  inline void gemv(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transA, long m, \
                   long n, T alpha, const T* a, long lda, const T* x, \
                   long incx, T beta, T* y, long incy) { \
    XI(P, gemv)(layout, transA, m, n, alpha, a, lda, x, incx, beta, y, incy); \
  } \
  inline void ger(CBLAS_LAYOUT layout, long m, long n, T alpha, \
                  const T* x, long incx, const T* y, long incy, \
                  T* a, long lda) { \
    XI(P, ger)(layout, m, n, alpha, x, incx, y, incy, a, lda); \
  } \
  inline void syr(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, long n, \
                  T alpha, const T* x, long incx, T* a, long lda) { \
    XI(P, syr)(layout, uplo, n, alpha, x, incx, a, lda); \
  } \
  inline void syr2(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, long n, \
                   T alpha, const T* x, long incx, const T* y, long incy,\
                   T* a, long lda) { \
    XI(P, syr2)(layout, uplo, n, alpha, x, incx, y, incy, a, lda); \
  } \
  inline void trmv(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, \
                   CBLAS_TRANSPOSE transA, CBLAS_DIAG diag, long n, \
                   const T* a, long lda, T* x, long incx) { \
    XI(P, trmv)(layout, uplo, transA, diag, n, a, lda, x, incx); \
  } \
  inline void trsv(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, \
                   CBLAS_TRANSPOSE transA, CBLAS_DIAG diag, long n, \
                   const T* a, long lda, T* x, long incx) { \
    XI(P, trsv)(layout, uplo, transA, diag, n, a, lda, x, incx); \
  } \
  inline void gemm(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transA, \
                   CBLAS_TRANSPOSE transB, long m, long n, long k, \
                   T alpha, const T* a, long lda, const T* b, long ldb, \
                   T beta, T* c, long ldc) { \
    XI(P, gemm)(layout, transA, transB, m, n, k, alpha, a, lda, b, ldb, beta, \
                c, ldc); \
  } \
  inline void symm(CBLAS_LAYOUT layout, CBLAS_SIDE side, CBLAS_UPLO uplo, \
                   long m, long n, T alpha, const T* a, long lda, \
                   const T* b, long ldb, T beta, T* c, long ldc) { \
    XI(P, symm)(layout, side, uplo, m, n, alpha, a, lda, b, ldb, beta, \
                c, ldc);  \
  } \
  inline void syrk(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, \
                   CBLAS_TRANSPOSE transA, long n, long k, T alpha, \
                   const T* a, long lda, T beta, T* c, long ldc) { \
    XI(P, syrk)(layout, uplo, transA, n, k, alpha, a, lda, beta, c, ldc); \
  } \
  inline void syr2k(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, \
                    CBLAS_TRANSPOSE trans, long n, long k, T alpha, \
                    const T* a, long lda, const T* b, long ldb, \
                    T beta, T* c, long ldc) { \
    XI(P, syr2k)(layout, uplo, trans, n, k, alpha, a, lda, b, ldb, beta, \
                 c, ldc); \
  } \
  inline void trmm(CBLAS_LAYOUT layout, CBLAS_SIDE side, CBLAS_UPLO uplo, \
                   CBLAS_TRANSPOSE transA, CBLAS_DIAG diag, long m, \
                   long n, T alpha, const T* a, long lda, T* b, long ldb) { \
    XI(P, trmm)(layout, side, uplo, transA, diag, m, n, alpha, a, lda, \
                b, ldb); \
  } \
  inline void trsm(CBLAS_LAYOUT layout, CBLAS_SIDE side, CBLAS_UPLO uplo, \
                   CBLAS_TRANSPOSE transA, CBLAS_DIAG diag, long m, \
                   long n, T alpha, const T* a, long lda, T* b, long ldb) { \
    XI(P, trsm)(layout, side, uplo, transA, diag, m, n, alpha, a, lda, \
                b, ldb); \
  }

DEFINE_OPS(float, s)
DEFINE_OPS(double, d)

inline float sdsdot(long n, float sb, const float* x, long incx,
                    const float* y, long incy) {
  return cblas_sdsdot(n, sb, x, incx, y, incy);
}

inline double dsdot(long n, const float* x, long incx,
                    const float* y, long incy) {
  return cblas_dsdot(n, x, incx, y, incy);
}

#undef DEFINE_OPS
#undef XI_I
#undef XI

}}}}  // namespaces

#endif /* DEEPLEARNING_TORCH_BLAS_H_ */


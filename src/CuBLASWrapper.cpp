// Copyright 2004-present Facebook. All Rights Reserved.

#include "CuBLASWrapper.h"

#include "cuda/DeviceTensor.cuh"
#include "THCTensor.h"
#include "BLASParameters.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <folly/ScopeGuard.h>
#include <glog/logging.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <vector>

using namespace std;
using namespace facebook::cuda;

namespace facebook { namespace deeplearning { namespace torch {

#define LOG_TARGET VLOG(3)

namespace {
const float kZero = 0.0f;
const float kOne = 1.0f;
const cuFloatComplex kZeroComplex = make_cuComplex(0.0f, 0.0f);
const cuFloatComplex kOneComplex = make_cuComplex(1.0f, 0.0f);
}


template <int Dim>
void transpose(const DeviceTensor<float, Dim>& in,
               DeviceTensor<float, Dim>& out,
               int sep,
               bool asComplex,
               cublasHandle_t handle,
               cudaStream_t stream) {
  cublasHandle_t localHandle;
  if (!handle) {
    cublasCreate(&localHandle);
  } else {
    localHandle = handle;
  }
  SCOPE_EXIT {
    if (!handle) {
      cublasDestroy(localHandle);
    }
  };

  // Only works on non-padded tensors since it is calling CUBLAS under the
  // hood. Could support padded along 1 dimension if needed.
  for (int i = 0; i < Dim; ++i) {
    CHECK_EQ(true, in.isContiguousDim(i)) << "Not contiguous dim = " << i;
    CHECK_EQ(true, out.isContiguousDim(i)) << "Not contiguous dim = " << i;
  }
  for (int i = 0; i < Dim; ++i) {
    CHECK_EQ(in.getSize(i), out.getSize(i)) <<
      "Not eq dim = " << i << " in = " << in << " out = " << out;
  }

  int rows = 1;
  for (int i = 0; i < sep; ++i) {
    rows *= in.getSize(i);
  }

  int cols = 1;
  for (int i = sep; i < Dim; ++i) {
    cols *= in.getSize(i);
  }

  if (stream) {
    cublasSetStream(localHandle, stream);
  }

  // As per cublas documentation:
  // For in-place mode, if C = A, ldc = lda and transa = CUBLAS_OP_N. If C =
  // B, ldc = ldb and transb = CUBLAS_OP_N. If the user does not meet above
  // requirements, CUBLAS_STATUS_INVALID_VALUE is returned.
  //
  // What this means is that in-place transpose is not supported.
  CHECK_NE(in.data(), out.data());

  cublasStatus_t res;
  if (asComplex) {
    // Must at least ensure cols is even
    CHECK_EQ(0, (cols & 1));
    cols >>= 1;
    // Root out alignment issues
    CHECK_EQ(0, (long)(in.template dataAs<cuFloatComplex>()) %
             sizeof(float2));
    CHECK_EQ(0, (long)(out.template dataAs<cuFloatComplex>()) %
             sizeof(float2));
    res = cublasCgeam(localHandle,
                      CUBLAS_OP_T,
                      CUBLAS_OP_N,
                      rows,
                      cols,
                      &kOneComplex,
                      in.template dataAs<cuFloatComplex>(),
                      cols,
                      &kZeroComplex,
                      nullptr,
                      rows,
                      out.template dataAs<cuFloatComplex>(),
                      rows);
  } else {
    res = cublasSgeam(localHandle,
                      CUBLAS_OP_T,
                      CUBLAS_OP_N,
                      rows,
                      cols,
                      &kOne,
                      in.data(),
                      cols,
                      &kZero,
                      nullptr,
                      rows,
                      out.data(),
                      rows);
  }
  CHECK_EQ(CUBLAS_STATUS_SUCCESS, res);

  // Permute the sizes to keep the CudaTensor consistent.
  // This only works because all dims are contiguous.
  std::vector<int> permDims;
  permDims.reserve(Dim);
  if (!asComplex) {
    // Non-complex case is easy
    for (int i = sep; i < Dim; ++i) {
      permDims.push_back(i);
    }
    for (int i = 0; i < sep; ++i) {
      permDims.push_back(i);
    }
  } else {
    // Complex case is trickier since it is float[2] that must stay in
    // horizontal order whatever happens
    for (int i = sep; i < Dim - 1; ++i) {
      permDims.push_back(i);
    }
    for (int i = 0; i < sep; ++i) {
      permDims.push_back(i);
    }
    permDims.push_back(Dim - 1);
  }

  out.permuteDims(permDims);

  THCudaCheck(cudaGetLastError());
  CHECK_EQ(CUBLAS_STATUS_SUCCESS, res);
}

template <int Dim>
void transposeAsComplex(const DeviceTensor<float, Dim>& in,
                        DeviceTensor<float, Dim>& out,
                        int sep,
                        cublasHandle_t handle,
                        cudaStream_t stream) {
  transpose<Dim>(in, out, sep, true, handle, stream);
}

#define TRANSPOSE_INSTANTIATION(DIM)                                    \
  template void transpose<DIM>(const DeviceTensor<float, DIM>& in,      \
                               DeviceTensor<float, DIM>& out,           \
                               int sep,                                 \
                               bool asComplex,                          \
                               cublasHandle_t handle,                   \
                               cudaStream_t stream);

#define TRANSPOSE_AS_COMPLEX_INSTANTIATION(DIM)                         \
  template void transposeAsComplex<DIM>(const DeviceTensor<float, DIM>& in, \
                                        DeviceTensor<float, DIM>& out,  \
                                        int sep,                        \
                                        cublasHandle_t handle,          \
                                        cudaStream_t stream);

TRANSPOSE_INSTANTIATION(2);
TRANSPOSE_INSTANTIATION(3);
TRANSPOSE_INSTANTIATION(4);
TRANSPOSE_INSTANTIATION(5);
TRANSPOSE_AS_COMPLEX_INSTANTIATION(2);
TRANSPOSE_AS_COMPLEX_INSTANTIATION(3);
TRANSPOSE_AS_COMPLEX_INSTANTIATION(4);
TRANSPOSE_AS_COMPLEX_INSTANTIATION(5);




namespace {

template <typename T>
T getCyclic(const std::vector<T>& list, unsigned int i) {
  if (list.size() == 0) {
    return nullptr;
  }
  return list[i % list.size()];
}

} // namespace anon

template <int Dim>
void matmult(DeviceTensor<float, Dim>& C,
             const DeviceTensor<float, Dim>& A,
             const DeviceTensor<float, Dim>& B,
             const BLASParameters& params) {
  cublasHandle_t localHandle;
  if (params.handles.size() == 0) {
    cublasCreate(&localHandle);
  } else {
    localHandle = getCyclic(params.handles, params.resourceIndex);
  }
  SCOPE_EXIT {
    if (params.handles.size() == 0) {
      cublasDestroy(localHandle);
    }
  };

  if (params.streams.size() > 0) {
    cublasSetStream(localHandle,
                    getCyclic(params.streams, params.resourceIndex));
  }

  LOG_TARGET << "Matmult single C (" << C << ") <- A (" << A <<
    ") * B (" << B << ")";
  LOG_TARGET << params;

  auto fastestVaryingA = A.getStride(0) / A.getStride(1);
  auto fastestVaryingB = B.getStride(0) / B.getStride(1);
  auto fastestVaryingC = C.getStride(0) / C.getStride(1);
  if (params.asComplex) {
    CHECK_EQ(3, Dim);
    CHECK_EQ(2, A.getSize(2));
    CHECK_EQ(2, B.getSize(2));
    CHECK_EQ(2, C.getSize(2));

    // In column major this is C' = B' * A'
    auto numReductions = (params.transposeB == CUBLAS_OP_N) ?
      B.getSize(0) : B.getSize(1);
    CHECK_EQ(numReductions,
             (params.transposeA == CUBLAS_OP_N) ? A.getSize(1) : A.getSize(0));
    auto scale = make_cuComplex(params.scaleRe, params.scaleIm);
    auto res = cublasCgemm(localHandle,
                           params.transposeB,
                           params.transposeA,
                           C.getSize(1),
                           C.getSize(0),
                           numReductions,
                           &scale,
                           B.template dataAs<cuFloatComplex>(),
                           fastestVaryingB,
                           A.template dataAs<cuFloatComplex>(),
                           fastestVaryingA,
                           (params.accumulate) ? &kOneComplex : &kZeroComplex,
                           C.template dataAs<cuFloatComplex>(),
                           fastestVaryingC
                          );
    CHECK_EQ(CUBLAS_STATUS_SUCCESS, res);
  } else {
    CHECK_EQ(2, Dim);
    // In column major this is C'nxm <- B'nxk * A'kxm
    auto numReductions = (params.transposeB == CUBLAS_OP_N) ?
      B.getSize(0) : B.getSize(1);
    CHECK_EQ(numReductions,
             (params.transposeA == CUBLAS_OP_N) ? A.getSize(1) : A.getSize(0));
    auto res = cublasSgemm(localHandle,
                           params.transposeB,
                           params.transposeA,
                           C.getSize(1),
                           C.getSize(0),
                           numReductions,
                           &params.scaleRe,
                           B.template data(),
                           fastestVaryingB,
                           A.template data(),
                           fastestVaryingA,
                           (params.accumulate) ? &kOne : &kZero,
                           C.template data(),
                           fastestVaryingC
                          );
    CHECK_EQ(CUBLAS_STATUS_SUCCESS, res);
  }
}

// Making it templated for automatic deduction but is only valid for Dim == 2.
// Cannot static check, perform a dynamic check.
// This avoids yet another level of structs to encapsulate a specialization.
template <int Dim>
void matmultBatchedLeaf(thrust::host_vector<float*>& CPtrVec,
                        thrust::host_vector<const float*>& APtrVec,
                        thrust::host_vector<const float*>& BPtrVec,
                        const DeviceTensor<float, Dim>& modelC,
                        const DeviceTensor<float, Dim>& modelA,
                        const DeviceTensor<float, Dim>& modelB,
                        const BLASParameters& params) {
  CHECK(!params.asComplex);
  CHECK_EQ(2, Dim);

  cublasHandle_t localHandle;
  if (params.handles.size() == 0) {
    cublasCreate(&localHandle);
  } else {
    localHandle = getCyclic(params.handles, params.resourceIndex);
  }
  SCOPE_EXIT {
    if (params.handles.size() == 0) {
      cublasDestroy(localHandle);
    }
  };

  if (params.streams.size() > 0) {
    cublasSetStream(localHandle,
                    getCyclic(params.streams, params.resourceIndex));
  }

  LOG_TARGET << "Matmult batched (" << CPtrVec.size() << ") from ptrs C (" <<
    modelC << ") <- A (" << modelA << ") * B (" << modelB << ")";
  LOG_TARGET << params;

  // This is blocking, make it a buffer if perf sensitive and multi-GPU from
  // single lua thread.
  thrust::device_vector<const float*> APtrDevice = APtrVec;
  thrust::device_vector<const float*> BPtrDevice = BPtrVec;
  thrust::device_vector<float*> CPtrDevice = CPtrVec;

  auto fastestVaryingA = modelA.getStride(0) / modelA.getStride(1);
  auto fastestVaryingB = modelB.getStride(0) / modelB.getStride(1);
  auto fastestVaryingC = modelC.getStride(0) / modelC.getStride(1);

  // In column major this is C' = B' * A'
  auto numReductions = (params.transposeB == CUBLAS_OP_N) ?
    modelB.getSize(0) : modelB.getSize(1);
  CHECK_EQ(numReductions,
           (params.transposeA == CUBLAS_OP_N) ? modelA.getSize(1) :
           modelA.getSize(0));
  auto res =
    cublasSgemmBatched(localHandle,
                       params.transposeB,
                       params.transposeA,
                       modelC.getSize(1),
                       modelC.getSize(0),
                       numReductions,
                       &params.scaleRe,
                       thrust::raw_pointer_cast(&BPtrDevice[0]),
                       fastestVaryingB,
                       thrust::raw_pointer_cast(&APtrDevice[0]),
                       fastestVaryingA,
                       (params.accumulate) ? &kOne : &kZero,
                       thrust::raw_pointer_cast(&CPtrDevice[0]),
                       fastestVaryingC,
                       CPtrVec.size()
                      );
  CHECK_EQ(CUBLAS_STATUS_SUCCESS, res);
}

// Making it templated for automatic deduction but is only valid for Dim == 3.
// Cannot static check, perform a dynamic check.
// This avoids yet another level of structs to encapsulate a specialization.
template <int Dim>
void matmultBatchedLeaf(thrust::host_vector<cuFloatComplex*>& CPtrVec,
                        thrust::host_vector<const cuFloatComplex*>& APtrVec,
                        thrust::host_vector<const cuFloatComplex*>& BPtrVec,
                        const DeviceTensor<float, Dim>& modelC,
                        const DeviceTensor<float, Dim>& modelA,
                        const DeviceTensor<float, Dim>& modelB,
                        const BLASParameters& params) {
  CHECK_EQ(3, Dim);
  CHECK(params.asComplex);
  CHECK_EQ(2, modelA.getSize(2));
  CHECK_EQ(2, modelB.getSize(2));
  CHECK_EQ(2, modelC.getSize(2));

  cublasHandle_t localHandle;
  if (params.handles.size() == 0) {
    cublasCreate(&localHandle);
  } else {
    localHandle = getCyclic(params.handles, params.resourceIndex);
  }
  SCOPE_EXIT {
    if (params.handles.size() == 0) {
      cublasDestroy(localHandle);
    }
  };

  if (params.streams.size() > 0) {
    cublasSetStream(localHandle,
                    getCyclic(params.streams, params.resourceIndex));
  }

  LOG_TARGET << "Matmult batched (" << CPtrVec.size() << ") from ptrs C (" <<
    modelC << ") <- A (" << modelA << ") * B (" << modelB << ")";
  LOG_TARGET << params;

  // This is blocking, make it a buffer if perf sensitive and multi-GPU from
  // single lua thread.
  // Either can write our own buffer or see if it is possible to have a
  // static monotonic array on the device side.
  thrust::device_vector<const cuFloatComplex*> APtrDevice = APtrVec;
  thrust::device_vector<const cuFloatComplex*> BPtrDevice = BPtrVec;
  thrust::device_vector<cuFloatComplex*> CPtrDevice = CPtrVec;

  auto fastestVaryingA = modelA.getStride(0) / modelA.getStride(1);
  auto fastestVaryingB = modelB.getStride(0) / modelB.getStride(1);
  auto fastestVaryingC = modelC.getStride(0) / modelC.getStride(1);

  // In column major this is C' = B' * A'
  auto numReductions = (params.transposeB == CUBLAS_OP_N) ?
    modelB.getSize(0) : modelB.getSize(1);
  CHECK_EQ(numReductions,
           (params.transposeA == CUBLAS_OP_N) ? modelA.getSize(1) :
           modelA.getSize(0));
  auto scale = make_cuComplex(params.scaleRe, params.scaleIm);
  auto res =
    cublasCgemmBatched(localHandle,
                       params.transposeB,
                       params.transposeA,
                       modelC.getSize(1),
                       modelC.getSize(0),
                       numReductions,
                       &scale,
                       thrust::raw_pointer_cast(&BPtrDevice[0]),
                       fastestVaryingB,
                       thrust::raw_pointer_cast(&APtrDevice[0]),
                       fastestVaryingA,
                       (params.accumulate) ? &kOneComplex : &kZeroComplex,
                       thrust::raw_pointer_cast(&CPtrDevice[0]),
                       fastestVaryingC,
                       CPtrVec.size()
                      );
  CHECK_EQ(CUBLAS_STATUS_SUCCESS, res);
}

namespace {

template <int Dim>
void matmultBatchedLeaf(DeviceTensor<float, Dim>& C,
                        DeviceTensor<float, Dim>& A,
                        DeviceTensor<float, Dim>& B,
                        const BLASParameters& params) {
  static_assert(Dim >= 3, "");
  if (params.asComplex) {
    CHECK_EQ(4, Dim);
    CHECK_EQ(2, A.getSize(3));
    CHECK_EQ(2, B.getSize(3));
    CHECK_EQ(2, C.getSize(3));
    thrust::host_vector<const cuFloatComplex*> APtrVec;
    thrust::host_vector<const cuFloatComplex*> BPtrVec;
    thrust::host_vector<cuFloatComplex*> CPtrVec;

    for (int batch = 0; batch < C.getSize(0); ++batch) {
      APtrVec.push_back(A[batch * params.batchStepA].template
                        dataAs<cuFloatComplex>());
      BPtrVec.push_back(B[batch * params.batchStepB].template
                        dataAs<cuFloatComplex>());
      CPtrVec.push_back(C[batch * params.batchStepC].template
                        dataAs<cuFloatComplex>());
    }

    matmultBatchedLeaf(CPtrVec,
                       APtrVec,
                       BPtrVec,
                       C.template view<Dim - 1>(),
                       A.template view<Dim - 1>(),
                       B.template view<Dim - 1>(),
                       params);
  } else {
    CHECK_EQ(3, Dim);
    thrust::host_vector<const float*> APtrVec;
    thrust::host_vector<const float*> BPtrVec;
    thrust::host_vector<float*> CPtrVec;

    for (int batch = 0; batch < C.getSize(0); ++batch) {
      APtrVec.push_back(A[batch * params.batchStepA].data());
      BPtrVec.push_back(B[batch * params.batchStepB].data());
      CPtrVec.push_back(C[batch * params.batchStepC].data());
    }

    matmultBatchedLeaf(CPtrVec,
                       APtrVec,
                       BPtrVec,
                       C.template view<Dim - 1>(),
                       A.template view<Dim - 1>(),
                       B.template view<Dim - 1>(),
                       params);
  }
}

// Dowcast version
template <int Dim, int NewDims>
struct matmultBatchedStruct {
  void run(DeviceTensor<float, Dim>& C,
           DeviceTensor<float, Dim>& A,
           DeviceTensor<float, Dim>& B,
           const BLASParameters& params) {
    static_assert(NewDims > 0, "");
    static_assert(NewDims < Dim, "");
    auto Cd = C.template downcastOuter<NewDims>();
    auto Ad = A.template downcastOuter<NewDims>();
    auto Bd = B.template downcastOuter<NewDims>();
    matmultBatchedLeaf(Cd, Ad, Bd, params);
  }
};

// No downcastOuter version
template <int Dim>
struct matmultBatchedStruct<Dim, Dim> {
  void run(DeviceTensor<float, Dim>& C,
           DeviceTensor<float, Dim>& A,
           DeviceTensor<float, Dim>& B,
           const BLASParameters& params) {
    matmultBatchedLeaf(C, A, B, params);
  }
};

#define BATCHEDMM_TAIL_INSTANTIATION(DIM1, DIM2)                        \
  template <>                                                           \
  struct matmultBatchedStruct<DIM1, DIM2> {                             \
    void run(DeviceTensor<float, DIM1>& C,                                \
             DeviceTensor<float, DIM1>& A,                          \
             DeviceTensor<float, DIM1>& B,                          \
             const BLASParameters& params) {                            \
      throw invalid_argument("BatchedMM needs at least 3 dimensions");  \
    }                                                                   \
  }                                                                     \

// Statically catch cases that should not be called
BATCHEDMM_TAIL_INSTANTIATION(3, 2);
BATCHEDMM_TAIL_INSTANTIATION(2, 2);
BATCHEDMM_TAIL_INSTANTIATION(3, 1);
BATCHEDMM_TAIL_INSTANTIATION(2, 1);
BATCHEDMM_TAIL_INSTANTIATION(1, 1);
BATCHEDMM_TAIL_INSTANTIATION(3, 0);
BATCHEDMM_TAIL_INSTANTIATION(2, 0);
BATCHEDMM_TAIL_INSTANTIATION(1, 0);

} // namespace anon

void matmultBatched(thrust::host_vector<cuFloatComplex*>& CPtrVec,
                    thrust::host_vector<const cuFloatComplex*>& APtrVec,
                    thrust::host_vector<const cuFloatComplex*>& BPtrVec,
                    const DeviceTensor<float, 3>& modelC,
                    const DeviceTensor<float, 3>& modelA,
                    const DeviceTensor<float, 3>& modelB,
                    const BLASParameters& params) {
  return matmultBatchedLeaf(
    CPtrVec, APtrVec, BPtrVec, modelC, modelA, modelB, params);
}

void matmultBatched(thrust::host_vector<float*>& CPtrVec,
                    thrust::host_vector<const float*>& APtrVec,
                    thrust::host_vector<const float*>& BPtrVec,
                    const DeviceTensor<float, 2>& modelC,
                    const DeviceTensor<float, 2>& modelA,
                    const DeviceTensor<float, 2>& modelB,
                    const BLASParameters& params) {
  return matmultBatchedLeaf(
    CPtrVec, APtrVec, BPtrVec, modelC, modelA, modelB, params);
}

template <int Dim>
void matmultBatched(DeviceTensor<float, Dim>& C,
                    DeviceTensor<float, Dim>& A,
                    DeviceTensor<float, Dim>& B,
                    const BLASParameters& params) {
  // Only works on non-padded tensors since it is calling CUBLAS under the
  // hood. Could support some padding if needed.
  for (int i = 0; i < Dim; ++i) {
    CHECK_EQ(true, A.isContiguousDim(i)) << "Not contiguous dim = " << i;
    CHECK_EQ(true, B.isContiguousDim(i)) << "Not contiguous dim = " << i;
    CHECK_EQ(true, C.isContiguousDim(i)) << "Not contiguous dim = " << i;
  }

  LOG_TARGET << "Matmult iter C (" << C << ") <- A (" << A <<
    ") * B (" << B << ")";
  LOG_TARGET << params;

  switch (params.batchDims) {
    case 0:
      matmult<Dim>(C, A, B, params);
      break;
    case 1:
      matmultBatchedStruct<Dim, Dim>().run(C, A, B, params);
      break;
    case 2:
      matmultBatchedStruct<Dim, Dim - 1>().run(C, A, B, params);
      break;
    default:
      throw invalid_argument("At most 2 outer sequential dimensions supported");
  };
}

template <int Dim, int NewDims>
struct matmultIterStruct {
  void run(DeviceTensor<float, Dim>& C,
           DeviceTensor<float, Dim>& A,
           DeviceTensor<float, Dim>& B,
           const BLASParameters& p) {
    static_assert(NewDims > 0, "");
    static_assert(NewDims < Dim, "");
    auto Cd = C.template downcastOuter<NewDims>();
    auto Ad = A.template downcastOuter<NewDims>();
    auto Bd = B.template downcastOuter<NewDims>();

    BLASParameters params = p;
    for (int i = 0; i < Cd.getSize(0); ++i) {
      auto tA = Ad[i].template view();
      auto tB = Bd[i].template view();
      auto tC = Cd[i].template view();
      matmultBatched<NewDims - 1>(tC, tA, tB, params.withResourceIndex(i));
    }
  }
};

template <int Dim>
struct matmultIterStruct<Dim, Dim> {
  void run(DeviceTensor<float, Dim>& C,
           DeviceTensor<float, Dim>& A,
           DeviceTensor<float, Dim>& B,
           const BLASParameters& p) {
    BLASParameters params = p;
    for (int i = 0; i < C.getSize(0); ++i) {
      auto tA = A[i].template view();
      auto tB = B[i].template view();
      auto tC = C[i].template view();
      matmultBatched<Dim - 1>(tC, tA, tB, params.withResourceIndex(i));
    }
  }
};

#define ITERATEDMM_TAIL_INSTANTIATION(DIM1, DIM2)       \
  template <>                                           \
  struct matmultIterStruct<DIM1, DIM2> {                \
    void run(DeviceTensor<float, DIM1>& C,                \
             DeviceTensor<float, DIM1>& A,                \
             DeviceTensor<float, DIM1>& B,                \
             const BLASParameters& params) {            \
      CHECK(false) << "Should not be here";             \
    }                                                   \
  };

ITERATEDMM_TAIL_INSTANTIATION(3, 1);
ITERATEDMM_TAIL_INSTANTIATION(2, 1);
ITERATEDMM_TAIL_INSTANTIATION(1, 1);
ITERATEDMM_TAIL_INSTANTIATION(3, 0);
ITERATEDMM_TAIL_INSTANTIATION(2, 0);
ITERATEDMM_TAIL_INSTANTIATION(1, 0);


template <int Dim>
void matmultIter(DeviceTensor<float, Dim>& C,
                 DeviceTensor<float, Dim>& A,
                 DeviceTensor<float, Dim>& B,
                 const BLASParameters& params) {
  // Only works on non-padded tensors since it is calling CUBLAS under the
  // hood. Could support some padding if needed.
  for (int i = 0; i < Dim; ++i) {
    CHECK(A.isContiguousDim(i)) << "Not contiguous dim = " << i;
    CHECK(B.isContiguousDim(i)) << "Not contiguous dim = " << i;
    CHECK(C.isContiguousDim(i)) << "Not contiguous dim = " << i;
  }

  LOG_TARGET << "Matmult iter C (" << C << ") <- A (" << A <<
    ") * B (" << B << ")";
  LOG_TARGET << params;

  switch (params.iterDims) {
    case 0:
      matmultBatched<Dim>(C, A, B, params);
      break;
    case 1:
      matmultIterStruct<Dim, Dim>().run(C, A, B, params);
      break;
    case 2:
      matmultIterStruct<Dim, Dim - 1>().run(C, A, B, params);
      break;

    default:
      throw invalid_argument(
        "At most 2 outer sequential and 2 batch dimensions supported");
  };
}

#define MATMULT_ITER_INSTANTIATION(DIM)                                 \
  template void matmultIter<DIM>(DeviceTensor<float, DIM>& C,             \
                                 DeviceTensor<float, DIM>& A,             \
                                 DeviceTensor<float, DIM>& B,             \
                                 const BLASParameters& params);

MATMULT_ITER_INSTANTIATION(2);
MATMULT_ITER_INSTANTIATION(3);
MATMULT_ITER_INSTANTIATION(4);
MATMULT_ITER_INSTANTIATION(5);
MATMULT_ITER_INSTANTIATION(6); // 2 iter + 2 batch + 2 mxm dims

} } } // namespace

// Copyright 2004-present Facebook. All Rights Reserved.
#pragma once

#include "cuda/DeviceTensor.cuh"

#include "BLASParameters.h"

#include "cublas_v2.h"
#include <iostream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

namespace facebook { namespace deeplearning { namespace torch {

//
// This transposition wrapper implements quick device-side transpositions.
// Consider tensor dimensions are collapsed into a 2-D 'y'-by-'x'.
// The wrapper takes a sep integer and considers dimensions (0 .. sep - 1) as
// being collapsed to form the 'y' dimension. Dimensions (sep .. Dim - 1)
// are collapsed to form the 'x' dimension.
//
// The complex case is a bit trickier since Torch does not natively support
// complex numbers, we emulate them with float[2]. In that case, 'x' is
// special in that it has to be exactly [x/2][2] and the inner 2 floats can
// never be transposed.
//
// The invariant is that in and out are sized identically on entry and that
// out is permuted to account for the transposition on exit.
//
// This wrapper requires non-padded tensors since it calls CUBLAS
// under the hood. It could support padding along 1 dimension if needed.
//
template<int Dim>
void transpose(const cuda::DeviceTensor<float, Dim>& in,
               cuda::DeviceTensor<float, Dim>& out,
               int sep,
               bool asComplex = false,
               cublasHandle_t handle = NULL,
               cudaStream_t stream = NULL);

template<int Dim>
void transposeAsComplex(const cuda::DeviceTensor<float, Dim>& in,
                        cuda::DeviceTensor<float, Dim>& out,
                        int sep,
                        cublasHandle_t handle = NULL,
                        cudaStream_t stream = NULL);

// Single matmult, not batched, not iterated, complex or real
template<int Dim>
void matmult(cuda::DeviceTensor<float, Dim>& C,
             const cuda::DeviceTensor<float, Dim>& A,
             const cuda::DeviceTensor<float, Dim>& B,
             const BLASParameters& params);


// Batched matmult from device pointers and model tensors serve to derive
// problem sizes. This is exposed for convenience to perform fancier batched
// sgemm calls.
void matmultBatched(thrust::host_vector<cuFloatComplex*>& CPtrVec,
                    thrust::host_vector<const cuFloatComplex*>& APtrVec,
                    thrust::host_vector<const cuFloatComplex*>& BPtrVec,
                    const cuda::DeviceTensor<float, 3>& modelC,
                    const cuda::DeviceTensor<float, 3>& modelA,
                    const cuda::DeviceTensor<float, 3>& modelB,
                    const BLASParameters& params);

// Batched matmult from device pointers and model tensors serve to derive
// problem sizes. This is exposed for convenience to perform fancier batched
// sgemm calls.
void matmultBatched(thrust::host_vector<float*>& CPtrVec,
                    thrust::host_vector<const float*>& APtrVec,
                    thrust::host_vector<const float*>& BPtrVec,
                    const cuda::DeviceTensor<float, 2>& modelC,
                    const cuda::DeviceTensor<float, 2>& modelA,
                    const cuda::DeviceTensor<float, 2>& modelB,
                    const BLASParameters& params);

// Batched matmult, not iterated, complex or real
// batchDims are outermost dimensions of the tensor iterated in parallel
template<int Dim>
void matmultBatched(cuda::DeviceTensor<float, Dim>& C,
                    cuda::DeviceTensor<float, Dim>& A,
                    cuda::DeviceTensor<float, Dim>& B,
                    const BLASParameters& params);

// Iterated matmult, batch or not, complex or real
// iterDims are outermost dimensions of the tensor iterated sequentially
// batchDims are outermost dimensions of the tensor iterated in parallel
template<int Dim>
void matmultIter(cuda::DeviceTensor<float, Dim>& C,
                 cuda::DeviceTensor<float, Dim>& A,
                 cuda::DeviceTensor<float, Dim>& B,
                 const BLASParameters& params);

} } } // namespace

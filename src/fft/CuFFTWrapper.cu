// Copyright 2004-present Facebook. All Rights Reserved.

#include "CuFFTWrapper.cuh"

#include "cuda/DeviceTensor.cuh"
#include "THCTensor.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <glog/logging.h>
#include <math_constants.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/replace.h>
#include <thrust/functional.h>
#include <gflags/gflags.h>

DEFINE_bool(fft_verbose, false, "Dump meta information for the FFT wrapper");

using namespace facebook::cuda;

namespace facebook { namespace deeplearning { namespace torch {

struct CudaScaleFunctor {
  const float s;

  CudaScaleFunctor(float _s) : s(_s) {}

  // This forces compilation with nvcc and is the reason why this is a .cu file
  __host__ __device__ float operator()(const float& x) const {
    return s * x;
  }
};


// makeCuFFTPlan allows the decoupling of plan making from plan execution.
//
// Batch dimensions are in the sense of cufft batch: the outer NumBatch
// dimensions will be specified as data parallel for cufft, their storage must
// be evenly spaced. If this is not the case, break the batched fft call into
// smaller calls.
//
// WARNING: The cuFFT documentation leads one to believe that storage dimension
// specification is innermost to outermost:
// http://docs.nvidia.com/cuda/cufft/index.html#ixzz39WscFf1z
// int inembed[NRANK] = {IX, IY}; // storage dimensions of input data
// IN FACT, IY is the fastest varying dimension in their case.
// This is confusing and goes against the dim3 convention.
template <int NumBatch, int RealTensorDim>
cufftHandle makeCuFFTPlan(const DeviceTensor<float, RealTensorDim>& real,
                          const DeviceTensor<float, RealTensorDim + 1>& cplx,
                          FFTParameters params) {
  const int kFFTDim = RealTensorDim - NumBatch;
  DCHECK_GE(3, kFFTDim);

  for (int i = 1; i < NumBatch; ++i) {
    DCHECK_LT(0, real.getStride(i)) << "Negative stride unsupported";
    DCHECK_LT(0, cplx.getStride(i)) << "Negative stride unsupported";
    DCHECK_EQ(real.getSize(i),
             real.getStride(i - 1) / real.getStride(i))
      << " Non-contiguous real batch storage, break your batched ffts into"
      << " smaller batches";
    DCHECK_EQ(cplx.getSize(i),
             cplx.getStride(i - 1) / cplx.getStride(i))
      << " Non-contiguous cplx batch storage, break your batched ffts into"
      << " smaller batches";
  }

  int batchSize = 1;
  for (int i = 0; i < NumBatch; ++i) {
    batchSize *= cplx.getSize(i);
  }
  if (FLAGS_fft_verbose) {
    LOG(INFO) << "MakeFFTPlan forward = " << params.forwardFFT() <<
      " with Batch Size " << batchSize;
  }

  int inSizeArr[kFFTDim] ;
  for (int i = 0; i < kFFTDim; ++i) {
    // "Input" signal size is always the real size, cufft uses hermitian
    // symmetry based on this isize.
    inSizeArr[(kFFTDim - 1) - i] = real.getSize(RealTensorDim - 1 - i);
    if (FLAGS_fft_verbose) {
      LOG(INFO) << "Size @"  << (char)('X' + i) << " -> "
                << inSizeArr[(kFFTDim - 1) - i];
    }
  }

  const int strideReal = 1;
  int storageReal[RealTensorDim];
  int distanceBetweenBatchesReal = 1;
  {
    int j = (RealTensorDim - 1) - 1; // Skip stride of 1
    for (int i = 0; i < kFFTDim; ++i, --j) {
      DCHECK_LT(0, real.getStride(j + 1)) <<
        "Stride <= 0 implies gather or reversal semantics, don't know " <<
        "how to FFT this atm";
      storageReal[(kFFTDim - 1) - i] =
        real.getStride(j) / real.getStride(j + 1);
      distanceBetweenBatchesReal *= storageReal[(kFFTDim - 1) - i];
      if (FLAGS_fft_verbose) {
        LOG(INFO) << "Storage real @" <<
          (char)('X' + (kFFTDim - 1) - i) << " -> " <<
          storageReal[(kFFTDim - 1) - i] << " ( " <<
          real.getStride(j) << " / " << real.getStride(j + 1) << ")";
      }
    }
  }
  if (FLAGS_fft_verbose) {
    LOG(INFO) << "distanceBetweenBatchesReal " << distanceBetweenBatchesReal;
  }

  // strides and distance is in terms of cplx entries (i.e. float[2])
  const int strideCplx = 1;
  int storageCplx[RealTensorDim];
  int distanceBetweenBatchesCplx = 1;
  {
    int j = (RealTensorDim - 1) - 1; // Skip strides of 1 and 2 (cplx float[2])
    for (int i = 0; i < kFFTDim; ++i, --j) {
      DCHECK_LT(0, cplx.getStride(j + 1)) <<
        "Stride <= 0 implies gather or reversal semantics, don't know " <<
        "how to FFT this atm";
      storageCplx[(kFFTDim - 1) - i] =
        cplx.getStride(j) / cplx.getStride(j + 1);
      distanceBetweenBatchesCplx *= storageCplx[(kFFTDim - 1) - i];
      if (FLAGS_fft_verbose) {
        LOG(INFO) << "Storage cplx @"  <<
          (char)('X' + (kFFTDim - 1) - i) << " -> " <<
          storageCplx[(kFFTDim - 1) - i] << " ( " <<
          cplx.getStride(j) <<
          " / " << cplx.getStride(j + 1) << " )";
      }
    }
  }

  if (FLAGS_fft_verbose) {
    LOG(INFO) << "distanceBtwBatchesCplx " << distanceBetweenBatchesCplx;
  }

  // Make sure cplx is really {float, float}
  cuda_static_assert(sizeof(float) == sizeof(cufftReal));
  cuda_static_assert(2 * sizeof(float) == sizeof(cufftComplex));

  cufftResult errFFT;
  cufftHandle plan;
  if (params.forwardFFT()) {
    errFFT = cufftPlanMany(&plan,
                           kFFTDim,
                           inSizeArr,
                           storageReal,
                           strideReal,
                           distanceBetweenBatchesReal,
                           storageCplx,
                           strideCplx,
                           distanceBetweenBatchesCplx,
                           CUFFT_R2C,
                           batchSize);
  } else {
    errFFT = cufftPlanMany(&plan,
                           kFFTDim,
                           inSizeArr,
                           storageCplx,
                           strideCplx,
                           distanceBetweenBatchesCplx,
                           storageReal,
                           strideReal,
                           distanceBetweenBatchesReal,
                           CUFFT_C2R,
                           batchSize);
  }
  if (errFFT != CUFFT_SUCCESS) {
    throw std::bad_alloc();
  }

  return plan;
}


template <int NumBatch, int RealTensorDim>
void fft(DeviceTensor<float, RealTensorDim>& real,
         DeviceTensor<float, RealTensorDim + 1>& cplx,
         FFTParameters params,
         cufftHandle* plan,
         cudaStream_t stream) {
  cufftHandle localPlan = (!plan) ?
    makeCuFFTPlan<NumBatch>(real, cplx, params) : *plan;

  if (stream) {
    cufftSetStream(localPlan, stream);
  }

  // At this point, must have a plan
  DCHECK_LT(0, (unsigned int)localPlan);

  cufftResult errFFT;
  if (params.forwardFFT()) {
    errFFT = cufftExecR2C(localPlan,
                          real.template dataAs<float>(),
                          cplx.template dataAs<cufftComplex>());
    if (errFFT != CUFFT_SUCCESS) {
      throw std::bad_alloc();
    }
    DCHECK_EQ(errFFT, CUFFT_SUCCESS);
  } else {
    errFFT = cufftExecC2R(localPlan,
                          cplx.template dataAs<cufftComplex>(),
                          real.template dataAs<cufftReal>());
    if (errFFT != CUFFT_SUCCESS) {
      throw std::bad_alloc();
    }
    DCHECK_EQ(errFFT, CUFFT_SUCCESS);

    if (params.normalizeFFT()) {
      // Normalize
      long size = 1;
      for (int i = NumBatch; i < RealTensorDim; ++i) {
        size *= real.getSize(i);
      }
      DCHECK_LT(0, size) << "Negative size not supported !";
      float val = 1 / (float)size;
      thrust::device_ptr<float> res(real.data());
      thrust::transform(res,
                        res + real.getSize(0) * real.getStride(0),
                        res,
                        CudaScaleFunctor(val));
    }
  }

  if (!plan) {
    // SCOPE_EXIT would be nice
    CHECK_EQ(CUFFT_SUCCESS, cufftDestroy(localPlan));
  }
}


template <int NumBatch>
void fft1d(DeviceTensor<float, NumBatch + 1>& real,
           DeviceTensor<float, NumBatch + 2>& cplx,
           FFTParameters params,
           cufftHandle* plan,
           cudaStream_t stream) {
  fft<NumBatch, NumBatch + 1>(real, cplx, params, plan, stream);
}

template <int NumBatch>
void fft2d(DeviceTensor<float, NumBatch + 2>& real,
           DeviceTensor<float, NumBatch + 3>& cplx,
           FFTParameters params,
           cufftHandle* plan,
           cudaStream_t stream) {
  fft<NumBatch, NumBatch + 2>(real, cplx, params, plan, stream);
}

template <int NumBatch>
void fft3d(DeviceTensor<float, NumBatch + 3>& real,
           DeviceTensor<float, NumBatch + 4>& cplx,
           FFTParameters params,
           cufftHandle* plan,
           cudaStream_t stream) {
  fft<NumBatch, NumBatch + 3>(real, cplx, params, plan, stream);
}

template void fft1d<2> (DeviceTensor<float, 3>& real,
                        DeviceTensor<float, 4>& cplx,
                        FFTParameters params,
                        cufftHandle* plan,
                        cudaStream_t stream);
template void fft1d<3> (DeviceTensor<float, 4>& real,
                        DeviceTensor<float, 5>& cplx,
                        FFTParameters params,
                        cufftHandle* plan,
                        cudaStream_t stream);
template void fft2d<2> (DeviceTensor<float, 4>& real,
                        DeviceTensor<float, 5>& cplx,
                        FFTParameters params,
                        cufftHandle* plan,
                        cudaStream_t stream);
template void fft3d<2> (DeviceTensor<float, 5>& real,
                        DeviceTensor<float, 6>& cplx,
                        FFTParameters params,
                        cufftHandle* plan,
                        cudaStream_t stream);

template void fft<1, 2>(DeviceTensor<float, 2>& real,
                        DeviceTensor<float, 3>& complex,
                        FFTParameters params,
                        cufftHandle* plan,
                        cudaStream_t stream);
template void fft<1, 3>(DeviceTensor<float, 3>& real,
                        DeviceTensor<float, 4>& complex,
                        FFTParameters params,
                        cufftHandle* plan,
                        cudaStream_t stream);
template void fft<1, 4>(DeviceTensor<float, 4>& real,
                        DeviceTensor<float, 5>& complex,
                        FFTParameters params,
                        cufftHandle* plan,
                        cudaStream_t stream);

template void fft<2, 3>(DeviceTensor<float, 3>& real,
                        DeviceTensor<float, 4>& complex,
                        FFTParameters params,
                        cufftHandle* plan,
                        cudaStream_t stream);
template void fft<2, 4>(DeviceTensor<float, 4>& real,
                        DeviceTensor<float, 5>& complex,
                        FFTParameters params,
                        cufftHandle* plan,
                        cudaStream_t stream);
template void fft<2, 5>(DeviceTensor<float, 5>& real,
                        DeviceTensor<float, 6>& complex,
                        FFTParameters params,
                        cufftHandle* plan,
                        cudaStream_t stream);

} } } // namespace

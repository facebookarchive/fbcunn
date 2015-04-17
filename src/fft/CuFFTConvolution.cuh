// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include "cuda/DeviceTensor.cuh"
#include "THC.h"
#include "THCTensor.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <glog/logging.h>

namespace facebook { namespace deeplearning { namespace torch {

struct ConvolutionPass {
 public:
  ConvolutionPass() : pass(kInvalid) {}

  explicit ConvolutionPass(unsigned int p) : pass(p) {
    DCHECK_GE(kSentinel, p);
  }

  static const unsigned int kInvalid = (unsigned int)-1;
  static const unsigned int kUpdateOutput = 0;
  static const unsigned int kAccGradParameters = 1;
  static const unsigned int kUpdateGradInput = 2;

  unsigned int pass;

  std::string toString() {
    DCHECK_GE(kSentinel, pass);
    if (pass == ConvolutionPass::kUpdateOutput) {
      return std::string("updateOutput     ");
    } else if (pass == ConvolutionPass::kUpdateGradInput) {
      return std::string("updateGradInput  ");
    } else if (pass == ConvolutionPass::kAccGradParameters) {
      return std::string("accGradParameters");
    }
    return std::string("Unknown");
  }

 private:
  static const unsigned int kSentinel = kUpdateGradInput;
};

struct CuFFTStrategy;

template<typename T>
T getCircular(std::vector<T>& v, int i) {
  if (v.size() == 0) {
    return NULL;
  }
  return v[i % v.size()];
}

struct CuFFTConvolution {
  // All 3 versions are essentially the same modulo parameters reordering.
  // Represent everything in terms of cgemm where C <- A . B
  // Complex conjugates and layouts follow from the description below:
  //
  // UpdateOutput
  // Cgemm input(y x b p) * filters(y x f p) (row-major formulation)
  // In other words, xy times o(b, f) <- conj(f(f, p)) . i(b, p)
  // However CUBLAS wants it in column-major:
  //   -> xy times o(f, b) <- f(p, f)* . i(p, b)
  //
  // AccGradParameters
  // Cgemm input(y x b p) * output(y x b f) (row-major formulation)
  // In other words, xy times f(f, p) <- conj(o(b, f)) . i(b, p)
  // However CUBLAS wants it in column-major:
  //   -> xy times f(p, f) <- i(p, b) . o(f, b)*
  //
  // UpdateGradInput
  // Cgemm filters(y x b f) * output(y x b f) (row-major formulation)
  // In other words, xy times i(b, p) <- f(f, p) . o(b, f)
  // However CUBLAS wants it in column-major:
  //   -> xy times i(p, b) <- f(p, f) . o(f, b)
  explicit CuFFTConvolution(ConvolutionPass pass) :
      strategy_(NULL),
      convPass_(pass),
      // These initializations to NULL are needed with nvcc ...
      fftPlanA_(NULL),
      fftPlanB_(NULL),
      fftPlanC_(NULL),
      fftPlanInverseA_(NULL),
      fftPlanInverseB_(NULL),
      cublasHandles_(),
      cudaStreams_(),
      cudaEvents_(),
      scale_(1.0f)
    {}

  // For each of input, output and buffers, one always need to pass at least
  // the real CudaTensor and a ComplexTH buffer to store the intermediate FFT.
  // The transpose buffer ComplexTrTH is also needed but will be cudaMalloc and
  // cudaFree on the fly if it is not specified.
  // The ComplexTrTH buffer has the same size as the ComplexTH buffer.
  CuFFTConvolution&
  withInputAndBuffers(THCState* state,
                      THCudaTensor* inputTH,
                      THCudaTensor* inputComplexTH,
                      THCudaTensor* inputComplexTrTH,
                      THCudaTensor* inputComplexBufferTH = NULL,
                      cufftHandle* plan = NULL,
                      cufftHandle* planInverse = NULL);

  CuFFTConvolution&
  withOutputAndBuffers(THCState* state,
                       THCudaTensor* outputTH,
                       THCudaTensor* outputComplexTH,
                       THCudaTensor* outputComplexTrTH,
                       THCudaTensor* outputComplexBufferTH = NULL,
                       cufftHandle* plan = NULL,
                       cufftHandle* planInverse = NULL);

  CuFFTConvolution&
  withFiltersAndBuffers(THCState* state,
                        THCudaTensor* filtersTH,
                        THCudaTensor* filtersComplexTH,
                        THCudaTensor* filtersComplexTrTH,
                        THCudaTensor* filtersComplexBufferTH = NULL,
                        cufftHandle* plan = NULL,
                        cufftHandle* planInverse = NULL);

  CuFFTConvolution&
  withCuBLASHandles(const std::vector<cublasHandle_t>& v) {
    cublasHandles_ = v;
    return *this;
  }

  CuFFTConvolution& withStreams(const std::vector<cudaStream_t>& v) {
    cudaStreams_ = v;
    return *this;
  }

  CuFFTConvolution& withEvents(const std::vector<cudaEvent_t>& v) {
    cudaEvents_ = v;
    return *this;
  }

  CuFFTConvolution& withScale(float s) {
    scale_ = s;
    return *this;
  }

  CuFFTConvolution& withStrategy(const CuFFTStrategy* s) {
    strategy_ = s;
    return *this;
  }

  cudaStream_t getStream(int i) {
    return getCircular(cudaStreams_, i);
  }

  cudaEvent_t getEvent(int i) {
    return getCircular(cudaEvents_, i);
  }

  void run();
  void reset();

  typedef cuda::DeviceTensor<float, 4> CudaRealTensor;
  typedef cuda::DeviceTensor<float, 5> CudaComplexTensor;

 private:
  CuFFTConvolution(const CuFFTConvolution&);
  CuFFTConvolution& operator=(const CuFFTConvolution&) const;

  void allToAllWait();
  void kWaitsOnAll(int k);
  void allWaitOnK(int k);
  void kWaitsOnL(int k, int l);
  void CuFFTConvolutionMxMBatch(const std::vector<cublasHandle_t>& handles);
  void CuFFTConvolutionMxMMany();
  void CuFFTConvolutionMxM();

  int mmReductionSize_;      // parameter   'k' in the Cgemm call
  int fastestVaryingSizeA_;  // parameter 'lda' in the Cgemm call
  int fastestVaryingSizeB_;  // parameter 'ldb' in the Cgemm call
  cublasOperation_t transA_;
  cublasOperation_t transB_;

  CudaRealTensor A_;
  CudaRealTensor B_;
  CudaRealTensor C_;
  CudaComplexTensor AComplex_;
  CudaComplexTensor BComplex_;
  CudaComplexTensor CComplex_;
  CudaComplexTensor AComplexT_;
  CudaComplexTensor BComplexT_;
  CudaComplexTensor CComplexT_;
  CudaComplexTensor AComplexBuffer_;
  CudaComplexTensor BComplexBuffer_;
  CudaComplexTensor CComplexBuffer_;

  // Normalize as part of MxM to avoid pass over iFFT data.
  // This is required only for CuFFT which does not perform any normalization
  // either in forward or inverse pass.
  cuComplex norm_;

  // C++11 tuple for automatic hash needed
  const CuFFTStrategy* strategy_;

  ConvolutionPass convPass_;

  cufftHandle* fftPlanA_;
  cufftHandle* fftPlanB_;
  cufftHandle* fftPlanC_;
  cufftHandle* fftPlanInverseA_;
  cufftHandle* fftPlanInverseB_;
  std::vector<cublasHandle_t> cublasHandles_;
  std::vector<cudaStream_t> cudaStreams_;
  std::vector<cudaEvent_t> cudaEvents_;

  float scale_;
};

} } } // namespace

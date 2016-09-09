// Copyright 2004-present Facebook. All Rights Reserved.

#include "src/DeviceTensorUtils.h"
#include "THCTensor.h"

#include "cuda/DeviceTensor.cuh"
#include "cuda/fbfft/FFTIteratedConvolution.cuh"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

using namespace facebook::cuda;

namespace facebook { namespace deeplearning { namespace torch {

typedef struct {
  THCudaTensor* tensor;
  int padL;
  int padU;
} TiledDeviceTensorFFI;

#define LOG_TARGET LOG(INFO)

#define INSTANTIATE_ITERATED_CONVOLUTION(DIM, FFT_SIZE)                 \
  if (THCudaTensor_nDimension(state, weight) == DIM &&                  \
      fftSize == FFT_SIZE) {                                            \
    thrust::host_vector<fbfft::detail::TiledDeviceTensor<float, DIM> >  \
      tiledInputs;                                                      \
    thrust::host_vector<fbfft::detail::TiledDeviceTensor<float, DIM> >  \
      tiledOutputs;                                                     \
    for (int i = 0; i < numTiles; ++i) {                                \
      DeviceTensor<float, DIM> ti(                                      \
        torchToDeviceTensor<float, DIM>(state, input[i].tensor));       \
      fbfft::detail::TiledDeviceTensor<float, DIM> inp(                 \
        ti,                                                             \
        input[i].padL,                                                  \
        input[i].padU);                                                 \
      /* TODO: emplace_back */                                          \
      tiledInputs.push_back(inp);                                       \
                                                                        \
      DeviceTensor<float, DIM> to(                                      \
        torchToDeviceTensor<float, DIM>(state, output[i].tensor));      \
      fbfft::detail::TiledDeviceTensor<float, DIM> out(                 \
        to,                                                             \
        output[i].padL,                                                 \
        output[i].padU);                                                \
      /* TODO: emplace_back */                                          \
      tiledOutputs.push_back(out);                                      \
    }                                                                   \
                                                                        \
    thrust::device_vector<fbfft::detail::TiledDeviceTensor<float, DIM> > \
      ins = tiledInputs;                                                \
    thrust::device_vector<fbfft::detail::TiledDeviceTensor<float, DIM> > \
      outs = tiledOutputs;                                              \
                                                                        \
    DeviceTensor<float, DIM> wei(                                       \
      torchToDeviceTensor<float, DIM>(state, weight));                  \
    bool res =                                                          \
      fbfft::detail::FFTIteratedConvolution<FFT_SIZE>(                  \
        thrust::raw_pointer_cast(&ins[0]),                              \
        thrust::raw_pointer_cast(&outs[0]),                             \
        wei,                                                            \
        pass,                                                           \
        scale,                                                          \
        batchSize,                                                      \
        ins.size(),                                                     \
        THCState_getCurrentStream(state));                              \
    if (!res) { THError("Error in iterated convolution"); }             \
  }

extern "C" void convolveIteratedFFI(THCState* state,
                                    TiledDeviceTensorFFI* input,
                                    THCudaTensor* weight,
                                    TiledDeviceTensorFFI* output,
                                    int numTiles,
                                    int fftSize,
                                    fbfft::detail::FFTConvolutionPassFFI pass,
                                    float scale) {
  // TODO: accGrad all on same stream, updateOutput / updateGradInput async
  int batchSize = THCudaTensor_size(state, input[0].tensor, 0);

  ////////////////////////////////////////////////////////
  // FFT of size 32
  ////////////////////////////////////////////////////////
  INSTANTIATE_ITERATED_CONVOLUTION(4, 32);

  ////////////////////////////////////////////////////////
  // FFT of size 16
  ////////////////////////////////////////////////////////
  INSTANTIATE_ITERATED_CONVOLUTION(4, 16);

  ////////////////////////////////////////////////////////
  // FFT of size 8
  ////////////////////////////////////////////////////////
  INSTANTIATE_ITERATED_CONVOLUTION(4, 8);
}

}}}

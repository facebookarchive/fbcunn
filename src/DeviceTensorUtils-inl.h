// Copyright 2004-present Facebook. All Rights Reserved.

#include <limits>
#include <stdexcept>

namespace facebook { namespace deeplearning { namespace torch {

template <typename T, int Dim,
          typename IndexT, template <typename U> class PtrTraits>
cuda::DeviceTensor<T, Dim, IndexT, PtrTraits>
torchToDeviceTensor(THCState* state, THCudaTensor* t) {
  if (Dim != THCudaTensor_nDimension(state, t)) {
    throw std::invalid_argument("THCudaTensor dimension mismatch");
  }

  // Determine the maximum offset into the tensor achievable; `IndexT`
  // must be smaller than this type in order to use it.
  long maxOffset = 0;
  IndexT sizes[Dim];
  IndexT strides[Dim];

  for (int i = 0; i < Dim; ++i) {
    long size = THCudaTensor_size(state, t, i);
    long stride = THCudaTensor_stride(state, t, i);

    maxOffset += (size - 1) * stride;

    sizes[i] = (IndexT) size;
    strides[i] = (IndexT) stride;
  }

  if (maxOffset > std::numeric_limits<IndexT>::max()) {
    throw std::invalid_argument(
      "THCudaTensor sizes too large for DeviceTensor conversion");
  }

  return cuda::DeviceTensor<T, Dim, IndexT, PtrTraits>(
    THCudaTensor_data(state, t), sizes, strides);
}

// Add a layer of SFINAE to support static_assert
template <typename T, int Dim, typename IndexT,
          template <typename U> class PtrTraits,
          int NewDim, bool B>
struct UpcastHostBuilderRoot {
  static cuda::DeviceTensor<T, NewDim, IndexT, PtrTraits>
  make(THCState* state, THCudaTensor* t);
};

template <typename T, int Dim, typename IndexT,
          template <typename U> class PtrTraits,
          int NewDim, bool B>
struct UpcastHostBuilder :
      UpcastHostBuilderRoot<T, Dim, IndexT, PtrTraits, NewDim, B> {
};

// Never instantiated SFINAE purposes only
template <typename T, int Dim, typename IndexT,
          template <typename U> class PtrTraits,
          int NewDim>
struct UpcastHostBuilder<T, Dim, IndexT, PtrTraits, NewDim, false> :
      UpcastHostBuilderRoot<T, Dim, IndexT, PtrTraits, NewDim, false> {
};

template <typename T, int Dim, typename IndexT,
          template <typename U> class PtrTraits,
          int NewDim>
struct UpcastHostBuilder<T, Dim, IndexT, PtrTraits, NewDim, true> :
      UpcastHostBuilderRoot<T, Dim, IndexT, PtrTraits, NewDim, true>  {
  static cuda::DeviceTensor<T, NewDim, IndexT, PtrTraits>
  make(THCState* state, THCudaTensor* t) {
    cuda_static_assert(NewDim > Dim);
    return torchToDeviceTensor<T, Dim, IndexT, PtrTraits>(state, t).
      template upcastOuter<NewDim>();
  }
};

// Add a layer of SFINAE to support static_assert
template <typename T, int Dim, typename IndexT,
          template <typename U> class PtrTraits,
          int NewDim, bool B>
struct DowncastHostBuilderRoot {
  static cuda::DeviceTensor<T, NewDim, IndexT, PtrTraits>
  make(THCState* state, THCudaTensor* t);
};

template <typename T, int Dim, typename IndexT,
          template <typename U> class PtrTraits,
          int NewDim, bool B>
struct DowncastHostBuilder :
      DowncastHostBuilderRoot<T, Dim, IndexT, PtrTraits, NewDim, B> {
};

// Never instantiated SFINAE purposes only
template <typename T, int Dim, typename IndexT,
          template <typename U> class PtrTraits,
          int NewDim>
struct DowncastHostBuilder<T, Dim, IndexT, PtrTraits, NewDim, false> :
      DowncastHostBuilderRoot<T, Dim, IndexT, PtrTraits, NewDim, false> {
};

template <typename T, int Dim, typename IndexT,
          template <typename U> class PtrTraits,
          int NewDim>
struct DowncastHostBuilder<T, Dim, IndexT, PtrTraits, NewDim, true> :
      DowncastHostBuilderRoot<T, Dim, IndexT, PtrTraits, NewDim, true>  {
  static cuda::DeviceTensor<T, NewDim, IndexT, PtrTraits>
  make(THCState* state, THCudaTensor* t) {
    cuda_static_assert(NewDim < Dim);
    return torchToDeviceTensor<T, Dim, IndexT, PtrTraits>(state, t).
      template downcastOuter<NewDim>();
  }
};

#define SWITCH_UNROLL_CUDA_CAST_FACTORY(i)                              \
  case i:                                                               \
  if (NewDim > i) {                                                     \
    return UpcastHostBuilder<T, i, IndexT,                              \
                             PtrTraits, NewDim, (NewDim > i)>::         \
      make(state, t);                                                   \
  } else if (NewDim == i) {                                             \
    return torchToDeviceTensor<T, NewDim, IndexT, PtrTraits>(state, t); \
  } else {                                                              \
    return DowncastHostBuilder<T, i, IndexT,                            \
                               PtrTraits, NewDim, (NewDim < i)>::       \
      make(state, t);                                                   \
  }                                                                     \
  /* break; */

template <typename T, int NewDim,
          typename IndexT, template <typename U> class PtrTraits>
cuda::DeviceTensor<T, NewDim, IndexT, PtrTraits>
torchToDeviceTensorCast(THCState* state, THCudaTensor* t) {
  switch (THCudaTensor_nDimension(state, t)) {
    SWITCH_UNROLL_CUDA_CAST_FACTORY(1);
    SWITCH_UNROLL_CUDA_CAST_FACTORY(2);
    SWITCH_UNROLL_CUDA_CAST_FACTORY(3);
    SWITCH_UNROLL_CUDA_CAST_FACTORY(4);
    SWITCH_UNROLL_CUDA_CAST_FACTORY(5);
    SWITCH_UNROLL_CUDA_CAST_FACTORY(6);
    SWITCH_UNROLL_CUDA_CAST_FACTORY(7);
    SWITCH_UNROLL_CUDA_CAST_FACTORY(8);
    SWITCH_UNROLL_CUDA_CAST_FACTORY(9);
    SWITCH_UNROLL_CUDA_CAST_FACTORY(10);
    default:
      ;
  }
  // Not implemented
  throw std::invalid_argument("DeviceTensor dimension size not supported");
}

} } }

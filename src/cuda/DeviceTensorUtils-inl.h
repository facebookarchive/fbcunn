// Copyright 2004-present Facebook. All Rights Reserved.

#include <limits>
#include <stdexcept>

namespace facebook { namespace deeplearning { namespace torch {

template <typename T, int Dim>
cuda::DeviceTensor<T, Dim> torchToDeviceTensor(THCudaTensor* t) {
  if (Dim != t->nDimension) {
    throw std::invalid_argument("THCudaTensor dimension mismatch");
  }

  int sizes[Dim];
  int strides[Dim];
  for (int i = 0; i < Dim; ++i) {
    // See t5239521
    if (t->size[i] > std::numeric_limits<int>::max()) {
      throw std::invalid_argument(
        "THCudaTensor sizes too large for DeviceTensor conversion");
    } else if (t->stride[i] > std::numeric_limits<int>::max()) {
      throw std::invalid_argument(
        "THCudaTensor strides too large for DeviceTensor conversion");
    }

    sizes[i] = (int) t->size[i];
    strides[i] = (int) t->stride[i];
  }

  return cuda::DeviceTensor<T, Dim>(
    t->storage->data + t->storageOffset, sizes, strides);
}

// Add a layer of SFINAE to support static_assert
template <typename T, int Dim, int NewDim, bool B>
struct UpcastHostBuilderRoot {
  static cuda::DeviceTensor<T, NewDim> make(THCudaTensor* tensor);
};

template <typename T, int Dim, int NewDim, bool B>
struct UpcastHostBuilder : UpcastHostBuilderRoot<T, Dim, NewDim, B> {
};

// Never instantiated SFINAE purposes only
template <typename T, int Dim, int NewDim>
struct UpcastHostBuilder<T, Dim, NewDim, false> :
      UpcastHostBuilderRoot<T, Dim, NewDim, false> {
};

template <typename T, int Dim, int NewDim>
struct UpcastHostBuilder<T, Dim, NewDim, true> :
      UpcastHostBuilderRoot<T, Dim, NewDim, true>  {
  static cuda::DeviceTensor<T, NewDim> make(THCudaTensor* tensor) {
    cuda_static_assert(NewDim > Dim);
    return torchToDeviceTensor<T, Dim>(tensor).template upcastOuter<NewDim>();
  }
};

// Add a layer of SFINAE to support static_assert
template <typename T, int Dim, int NewDim, bool B>
struct DowncastHostBuilderRoot  {
  static cuda::DeviceTensor<T, NewDim> make(THCudaTensor* tensor);
};

template <typename T, int Dim, int NewDim, bool B>
struct DowncastHostBuilder : DowncastHostBuilderRoot<T, Dim, NewDim, B> {
};

// Never instantiated SFINAE purposes only
template <typename T, int Dim, int NewDim>
struct DowncastHostBuilder<T, Dim, NewDim, false> :
      DowncastHostBuilderRoot<T, Dim, NewDim, false> {
};

template <typename T, int Dim, int NewDim>
struct DowncastHostBuilder<T, Dim, NewDim, true> :
      DowncastHostBuilderRoot<T, Dim, NewDim, true>  {
  static cuda::DeviceTensor<T, NewDim> make(THCudaTensor* tensor) {
    cuda_static_assert(NewDim < Dim);
    return torchToDeviceTensor<T, Dim>(tensor).template downcast<NewDim>();
  }
};

#define SWITCH_UNROLL_CUDA_CAST_FACTORY(i)                              \
  case i:                                                               \
    if (NewDim > i) {                                                   \
      return UpcastHostBuilder<T, i, NewDim, (NewDim > i)>::make(tensor); \
    } else if (NewDim == i) {                                           \
      return torchToDeviceTensor<T, NewDim>(tensor);                    \
    } else {                                                            \
      return DowncastHostBuilder<T, i, NewDim, (NewDim < i)>::make(tensor); \
    }                                                                   \
    /* break; */


template <typename T, int NewDim>
cuda::DeviceTensor<T, NewDim>
torchToDeviceTensorCast(THCudaTensor* tensor) {
  switch (tensor->nDimension) {
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
  throw std::invalid_argument("CastFactory input dims error");
}

} } }

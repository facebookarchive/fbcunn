// Copyright 2004-present Facebook. All Rights Reserved.
#include "cuda/DeviceTensor.cuh"
#include "torch/fb/fbcunn/src/DeviceTensorUtils.h"

#include "torch/fb/fbcunn/src/util/Misc.h"

#include <cuda.h>

using namespace facebook::cuda;
using namespace facebook::CUDAUtil;

namespace facebook { namespace deeplearning { namespace torch {

__global__ void testAssignment1dKernel(DeviceTensor<float, 1> tensor) {
  // Thread grid is already sized exactly for our tensor
  tensor[threadIdx.x] = threadIdx.x;
}

bool testAssignment1d(THCState* state, THCudaTensor* t) {
  DeviceTensor<float, 1> tensor =
    torchToDeviceTensor<float, 1>(state, t);

  const cudaDeviceProp& deviceProp = getDeviceProperties(0);

  if (deviceProp.maxThreadsDim[0] < tensor.getSize(0)) {
    // tensor too large to be covered exactly by threads in one block alone
    return false;
  }

  testAssignment1dKernel<<<1, tensor.getSize(0)>>>(tensor);

  return (cudaGetLastError() == cudaSuccess);
}

__global__ void testAssignment3dKernel(DeviceTensor<float, 3> tensor) {
  // Thread grid is already sized exactly for our tensor
  tensor[threadIdx.z][threadIdx.y][threadIdx.x] =
    tensor.getSize(0) * threadIdx.z +
    tensor.getSize(1) * threadIdx.y +
    tensor.getSize(2) * threadIdx.x;
}

bool testAssignment3d(THCState* state, THCudaTensor* t) {
  DeviceTensor<float, 3> tensor = torchToDeviceTensor<float, 3>(state, t);

  const cudaDeviceProp& deviceProp = getDeviceProperties(0);

  for (int i = 0; i < 3; ++i) {
    if (deviceProp.maxThreadsDim[i] < tensor.getSize(i)) {
      // tensor too large to be covered exactly by threads in one block alone
      return false;
    }
  }

  dim3 threadsPerBlock(tensor.getSize(2),
                       tensor.getSize(1),
                       tensor.getSize(0));
  testAssignment3dKernel<<<1, threadsPerBlock>>>(tensor);

  return (cudaGetLastError() == cudaSuccess);
}

template <int NewDim, int Dim>
bool verifyUpcast(DeviceTensor<float, NewDim> up,
                  DeviceTensor<float, Dim> orig) {
  int shift = NewDim - Dim;

  // Check extended dimensions size and stride
  for (int i = 0; i < shift; ++i) {
    if (up.getSize(i) != 1) {
      return false;
    } else if (up.getStride(i) !=
               orig.getStride(0) * orig.getSize(0)) {
      return false;
    }
  }

  // Check original dimensions size and stride
  for (int i = shift; i < NewDim; ++i) {
    if (up.getSize(i) != orig.getSize(i - shift)) {
      return false;
    } else if (up.getStride(i) != orig.getStride(i - shift)) {
      return false;
    }
  }

  return true;
}

bool testUpcast(THCState* state, THCudaTensor* t) {
  DeviceTensor<float, 3> tensor = torchToDeviceTensor<float, 3>(state, t);

  if (!verifyUpcast(tensor.upcastOuter<4>(), tensor)) {
    return false;
  } else if (!verifyUpcast(tensor.upcastOuter<5>(), tensor)) {
    return false;
  }

  return true;
}

bool testDowncastTo2d(THCState* state, THCudaTensor* t) {
  DeviceTensor<float, 3> tensor = torchToDeviceTensor<float, 3>(state, t);
  DeviceTensor<float, 2> downTensor = tensor.downcastOuter<2>();

  if (downTensor.getSize(0) !=
      tensor.getSize(0) * tensor.getSize(1)) {
    return false;
  } else if (downTensor.getStride(0) !=
             tensor.getSize(2) * tensor.getStride(2)) {
    return false;
  } else if (downTensor.getSize(1) !=
             tensor.getSize(2)) {
    return false;
  } else if (downTensor.getStride(1) !=
             tensor.getStride(2)) {
    return false;
  }

  return true;
}

bool testDowncastTo1d(THCState* state, THCudaTensor* t) {
  DeviceTensor<float, 3> tensor = torchToDeviceTensor<float, 3>(state, t);
  DeviceTensor<float, 1> downTensor = tensor.downcastOuter<1>();

  if (downTensor.getSize(0) !=
      tensor.getSize(0) * tensor.getSize(1) * tensor.getSize(2)) {
    return false;
  } else if (downTensor.getStride(0) !=
             tensor.getStride(2)) {
    return false;
  }

  return true;
}

__global__ void testDowncastWritesKernel(DeviceTensor<float, 1> tensor) {
  // Thread grid is already sized exactly for our tensor
  tensor[threadIdx.x] = 1.0f;
}

bool testDowncastWrites(THCState* state, THCudaTensor* t) {
  DeviceTensor<float, 3> tensor = torchToDeviceTensor<float, 3>(state, t);
  DeviceTensor<float, 1> downTensor = tensor.downcastOuter<1>();

  testDowncastWritesKernel<<<1, downTensor.getSize(0)>>>(downTensor);
  return (cudaGetLastError() == cudaSuccess);
}

} } } // namespace

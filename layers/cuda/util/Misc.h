// Copyright 2004-, Facebook, Inc. All Rights Reserved.

#pragma once

#include <cuda_runtime.h>

namespace facebook { namespace CUDAUtil {

int getDevice();

extern __attribute__((__noreturn__))
void throwCudaError(cudaError_t, const char* msg);

inline void
checkCudaError(cudaError_t error, const char* msg = 0) {
  if (error != cudaSuccess) {
    throwCudaError(error, msg);
  }
}


class OnDevice {
  int m_home;
 public:
  explicit OnDevice(int newDev) : m_home(getDevice()) {
    checkCudaError(cudaSetDevice(newDev));
  }

  ~OnDevice() {
    checkCudaError(cudaSetDevice(m_home));
  }
};

const cudaDeviceProp& getCurrentDeviceProperties();
const cudaDeviceProp& getDeviceProperties(int device);

cudaStream_t getComputeStream();
cudaStream_t getCopyStream();

} }

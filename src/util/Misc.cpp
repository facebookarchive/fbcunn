// Copyright 2004-, Facebook, Inc. All Rights Reserved.

#include "util/Misc.h"
#include <folly/Format.h>
#include <folly/Memory.h>
#include <mutex>
#include <unordered_map>

using namespace std;

namespace facebook { namespace CUDAUtil {

int getDevice() {
  int dev;
  checkCudaError(cudaGetDevice(&dev));
  return dev;
}

// Streams. We have an implicit model that async memory copies
// with send semantics happen on a dedicated, conventional stream
// per-device. The stream runs on the destination.
namespace {
mutex mtx;
unordered_map<int, cudaStream_t> deviceToCopyStream;
}

cudaStream_t getComputeStream() {
  // It would be nice to compute on non-default streams from time to time,
  // but there's a *lot* of code to change.
  return 0;
}

cudaStream_t getCopyStream() {
  unique_lock<mutex> own(mutex);
  auto dev = getDevice();
  auto row = deviceToCopyStream.find(dev);
  if (row == deviceToCopyStream.end()) {
    cudaStream_t& stream = deviceToCopyStream[dev];
    checkCudaError(cudaStreamCreate(&stream));
    return stream;
  }
  return row->second;
}

__attribute__((__noreturn__))
void throwCudaError(cudaError_t error, const char* msg) {
  auto string = msg ?
    folly::sformat("{}: CUDA error {} ({})", msg, int(error),
                   cudaGetErrorString(error)) :
      folly::sformat("CUDA error {} ({})", int(error),
                     cudaGetErrorString(error));
  throw std::runtime_error(string);
}

namespace {

struct DeviceProperties {
  DeviceProperties();
  int deviceCount = 0;
  std::unique_ptr<cudaDeviceProp[]> deviceProperties;
};

DeviceProperties::DeviceProperties() {
  auto err = cudaGetDeviceCount(&deviceCount);
  if (err == cudaErrorNoDevice) {
    deviceCount = 0;
  } else {
    checkCudaError(err, "cudaGetDeviceCount");
  }

  deviceProperties = folly::make_unique<cudaDeviceProp[]>(deviceCount);
  for (int i = 0; i < deviceCount; ++i) {
    checkCudaError(cudaGetDeviceProperties(&deviceProperties[i], i),
                   "cudaGetDeviceProperties");
  }
}

}  // namespace

const cudaDeviceProp& getCurrentDeviceProperties() {
  int device = 0;
  checkCudaError(cudaGetDevice(&device), "cudaGetDevice");

  return getDeviceProperties(device);
}

const cudaDeviceProp& getDeviceProperties(int device) {
  static DeviceProperties dprop;
  DCHECK(device >= 0 && device < dprop.deviceCount);
  return dprop.deviceProperties[device];
}

} }

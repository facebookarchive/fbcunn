// Copyright 2004-, Facebook, Inc. All Rights Reserved.

#include "src/util/Misc.h"

#include <folly/Format.h>
#include <folly/Memory.h>
#include <mutex>
#include <unordered_map>

using namespace std;

namespace facebook { namespace cuda {

cudaStream_t getComputeStream() {
  // It would be nice to compute on non-default streams from time to time,
  // but there's a *lot* of code to change.
  return 0;
}

[[noreturn]] void throwCudaError(cudaError_t error, const char* msg) {
  auto string = msg ?
    folly::sformat("{}: CUDA error {} ({})", msg, int(error),
                   cudaGetErrorString(error)) :
      folly::sformat("CUDA error {} ({})", int(error),
                     cudaGetErrorString(error));
  throw std::runtime_error(string);
}

} }

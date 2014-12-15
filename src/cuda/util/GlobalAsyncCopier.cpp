/**
 * Copyright 2014 Facebook
 * @author Tudor Bosman (tudorb@fb.com)
 */

#include "util/GlobalAsyncCopier.h"

#include <cstdlib>
#include <folly/Conv.h>
#include <folly/Memory.h>

#include "util/AsyncCopier.h"

using namespace facebook::CUDAUtil;

constexpr size_t kDefaultBufferSizeMB = 16;
const char* const kBufferSizeEnvVar = "FB_CUDA_ASYNC_COPIER_BUFFER_SIZE_MB";

std::unique_ptr<AsyncCopier> makeGlobalCopier() {
  size_t bufferSize = kDefaultBufferSizeMB;
  auto ptr = getenv(kBufferSizeEnvVar);
  if (ptr) {
    bufferSize = folly::to<size_t>(ptr);
  }

  return folly::make_unique<AsyncCopier>(bufferSize << 20);
}

extern "C" void fbCudaAsyncMemcpyHtoD(void* dest,
                                      const void* src,
                                      size_t size) {
  static auto copier = makeGlobalCopier();
  copier->copyHtoD(dest, src, size);
}

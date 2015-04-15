/**
 * Copyright 2014 Facebook
 * @author Tudor Bosman (tudorb@fb.com)
 */

#ifndef DEEPLEARNING_TORCH_CUDA_UTIL_ASYNCCOPIER_H_
#define DEEPLEARNING_TORCH_CUDA_UTIL_ASYNCCOPIER_H_

#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <vector>
#include <cuda_runtime_api.h>
#include <folly/Optional.h>
#include <folly/small_vector.h>

namespace facebook { namespace CUDAUtil {

class AsyncCopier {
 public:
  explicit AsyncCopier(size_t bufferSize);

  void copyHtoD(void* dest, const void* src, size_t size);

 private:
  class Deallocator {
   public:
    void operator()(uint8_t* ptr) const;
  };

  struct Event {
    explicit Event(int device);

    int device;
    folly::Optional<cudaEvent_t> event;
    ssize_t refCount;
  };

  struct AllocatedBlock {
    AllocatedBlock(size_t s, size_t l) : start(s), length(l) { }
    size_t start;
    size_t length;
    Event* event = nullptr;
  };

  static bool pollEvent(Event* event);  // returns true if completed
  static void waitEvent(Event* event);

  typedef folly::small_vector<AllocatedBlock, 2> RangeVec;
  RangeVec getRangesLocked() const;
  Event* getEventLocked();
  void releaseEventLocked(Event* event);

  const size_t bufferSize_;
  std::unique_ptr<uint8_t[], Deallocator> buffer_;

  std::mutex mutex_;
  std::vector<std::deque<Event>> events_;
  std::vector<std::vector<Event*>> freeEvents_;
  std::deque<AllocatedBlock> allocated_;
};

}}  // namespaces

#endif /* DEEPLEARNING_TORCH_CUDA_UTIL_ASYNCCOPIER_H_ */

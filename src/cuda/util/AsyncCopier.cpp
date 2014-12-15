/**
 * Copyright 2014 Facebook
 * @author Tudor Bosman (tudorb@fb.com)
 */

#include "util/AsyncCopier.h"
#include "util/Misc.h"
#include <exception>
#include <folly/Format.h>
#include <glog/logging.h>

namespace facebook { namespace CUDAUtil {

uint8_t* allocPageLocked(size_t size) {
  void* ptr;
  checkCudaError(cudaHostAlloc(&ptr, size, cudaHostAllocPortable),
                 "cudaHostAlloc");
  return static_cast<uint8_t*>(ptr);
}

void AsyncCopier::Deallocator::operator()(uint8_t* ptr) const {
  if (ptr) {
    cudaFreeHost(ptr);
  }
}

AsyncCopier::Event::Event(int d)
  : device(d),
    refCount(0) {
  event.emplace();
  checkCudaError(
      cudaEventCreateWithFlags(
          get_pointer(event), cudaEventDisableTiming | cudaEventBlockingSync),
      "cudaEventCreateWithFlags");
}

AsyncCopier::AsyncCopier(size_t bufferSize)
  : bufferSize_(bufferSize),
    buffer_(allocPageLocked(bufferSize)) {
  int deviceCount;
  checkCudaError(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount");
  events_.resize(deviceCount);
  freeEvents_.resize(deviceCount);
}

bool AsyncCopier::pollEvent(Event* event) {
  auto result = cudaEventQuery(*event->event);
  switch (result) {
  case cudaSuccess:
    VLOG(2) << "Poll event " << *event->event << ": ready";
    return true;
  case cudaErrorNotReady:
    VLOG(2) << "Poll event " << *event->event << ": not ready";
    return false;
  default:
    throwCudaError(result, "cudaEventQuery");
  }
}

void AsyncCopier::waitEvent(Event* event) {
  VLOG(2) << "Wait for event " << *event->event;
  checkCudaError(cudaEventSynchronize(*event->event), "cudaEventSynchronize");
}

auto AsyncCopier::getEventLocked() -> Event* {
  int device;
  checkCudaError(cudaGetDevice(&device), "cudaGetDevice");

  auto& events = events_[device];
  auto& freeEvents = freeEvents_[device];

  Event* ev;
  if (!freeEvents.empty()) {
    ev = freeEvents.back();
    freeEvents.pop_back();
    VLOG(2) << "Get free event " << *ev->event;
  } else {
    events.emplace_back(device);
    ev = &events.back();
    VLOG(2) << "Allocate new event " << *ev->event;
  }
  ++ev->refCount;
  return ev;
}

void AsyncCopier::releaseEventLocked(Event* ev) {
  if (--ev->refCount <= 0) {
    DCHECK_EQ(ev->refCount, 0);
    VLOG(2) << "Release event " << *ev->event;
    freeEvents_[ev->device].push_back(ev);
  }
}

// Return the unallocated ranges; at most two: one at the end of the
// buffer and one at the beginning.
auto AsyncCopier::getRangesLocked() const -> RangeVec {
  RangeVec ranges;
  if (allocated_.empty()) {
    ranges.emplace_back(0, bufferSize_);
  } else {
    auto& first = allocated_.front();
    auto& last = allocated_.back();
    auto start = first.start;
    auto end = last.start + last.length;

    if (start < end) {
      if (end < bufferSize_) {
        ranges.emplace_back(end, bufferSize_ - end);
      }
      if (start > 0) {
        ranges.emplace_back(0, start);
      }
    } else if (start > end) {
      ranges.emplace_back(end, start - end);
    }
  }
  DCHECK(ranges.size() <= 2);
  return ranges;
}

void AsyncCopier::copyHtoD(void* dest, const void* src, size_t size) {
  VLOG(1) << "copyHtoD " << size;
  auto pdest = static_cast<uint8_t*>(dest);
  auto psrc = static_cast<const uint8_t*>(src);

  unsigned int flags;
  auto err = cudaHostGetFlags(&flags, const_cast<void*>(src));
  if (err == cudaSuccess) {
    // Page-locked using cudaHostAlloc / cudaHostRegister, copy directly.
    checkCudaError(cudaMemcpyAsync(dest, src, size, cudaMemcpyHostToDevice),
                   "cudaMemcpyAsync");
    return;
  } else if (err != cudaErrorInvalidValue) {
    checkCudaError(err, "invalid return code from cudaMemcpyAsync");
  }
  cudaGetLastError();  // reset last error
  // This is dicey -- what if another kernel has completed with an error?
  // But there's nothing else we can do, as any cuda function may return an
  // error from a previous kernel launch.

  if (size > bufferSize_) {
    // Copy synchronously.
    checkCudaError(cudaMemcpy(dest, src, size, cudaMemcpyHostToDevice),
                   "cudaMemcpy");
    return;
  }

  Event* eventToWait = nullptr;

  auto copyRange = [this, &size, &pdest, &psrc] (AllocatedBlock& range) {
    size_t n = std::min(size, range.length);
    range.length = n;
    VLOG(1) << "Copy " << range.start << " + " << n;
    auto bufPtr = buffer_.get() + range.start;
    memcpy(bufPtr, psrc, n);
    checkCudaError(cudaMemcpyAsync(pdest, bufPtr, n, cudaMemcpyHostToDevice),
                   "cudaMemcpyAsync");
    pdest += n;
    psrc += n;
    size -= n;
    checkCudaError(cudaEventRecord(*range.event->event), "cudaEventRecord");
    allocated_.push_back(range);
  };

  for (;;) {
    {
      std::lock_guard<std::mutex> lock(mutex_);

      if (eventToWait) {
        releaseEventLocked(eventToWait);
        eventToWait = nullptr;
      }

      // Always reap
      while (!allocated_.empty() && pollEvent(allocated_.front().event)) {
        releaseEventLocked(allocated_.front().event);
        allocated_.pop_front();
      }

      auto ranges = getRangesLocked();
      if (!ranges.empty()) {
        auto ev = getEventLocked();
        for (auto it = ranges.begin(); size != 0 && it != ranges.end(); ++it) {
          auto& range = *it;
          ++ev->refCount;
          range.event = ev;
          copyRange(range);
        }
        releaseEventLocked(ev);
        if (size == 0) {
          break;
        }
      }
      // Sigh, we have to wait.
      eventToWait = allocated_.front().event;
      ++eventToWait->refCount;
    }

    DCHECK(eventToWait);
    VLOG(1) << "Waiting, remaining " << size;
    waitEvent(eventToWait);
  }
  VLOG(1) << "End copyHtoD";

  DCHECK(!eventToWait);
}

}}  // namespaces

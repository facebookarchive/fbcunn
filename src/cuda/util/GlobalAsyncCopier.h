/**
 * Copyright 2014 Facebook
 * @author Tudor Bosman (tudorb@fb.com)
 */

#ifndef DEEPLEARNING_TORCH_CUDA_UTIL_GLOBALASYNCCOPIER_H_
#define DEEPLEARNING_TORCH_CUDA_UTIL_GLOBALASYNCCOPIER_H_

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void fbCudaAsyncMemcpyHtoD(void* dest, const void* src, size_t size);

#ifdef __cplusplus
}
#endif

#endif /* DEEPLEARNING_TORCH_CUDA_UTIL_GLOBALASYNCCOPIER_H_ */

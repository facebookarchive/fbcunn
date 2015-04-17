// Copyright 2004-present Facebook. All Rights Reserved.
#include "CudaTensorUtils.h"
#include "THC.h"

using namespace std;
using namespace thpp;

void
CudaTensorDeleter::operator()(THCudaTensor* t) {
  THCudaTensor_free(state, t);
}

namespace facebook { namespace deeplearning { namespace torch {

unique_ptr<THCudaTensor, CudaTensorDeleter>
makeTHCudaTensorFull(THCState* state,
                     const vector<long>& sizes,
                     const folly::Optional<vector<long>>& strides) {
  if (sizes.empty()) {
    throw invalid_argument("must have 1 or more dimensions");
  } else if (strides && strides->size() != sizes.size()) {
    throw invalid_argument("sizes and strides must match");
  }

  long size = 1;
  if (!strides) {
    for (const auto s : sizes) {
      size *= s;
    }
  } else {
    size = strides->front() * sizes.front();
  }

  auto storage = THCudaStorage_newWithSize(state, size);

  auto tensor = THCudaTensor_new(state);
  tensor->storage = storage;
  tensor->storageOffset = 0;
  tensor->nDimension = sizes.size();

  tensor->size = (long*) THAlloc(sizeof(long) * sizes.size());
  memcpy(tensor->size, sizes.data(), sizeof(long) * sizes.size());

  tensor->stride = (long*) THAlloc(sizeof(long) * sizes.size());
  if (strides) {
    memcpy(tensor->stride, strides->data(), sizeof(long) * strides->size());
  } else {
    vector<long> tmpStrides(sizes.size());

    tmpStrides.back() = 1;
    for (int i = tmpStrides.size() - 2; i >= 0; --i) {
      tmpStrides[i] = tmpStrides[i + 1] * sizes[i + 1];
    }

    memcpy(tensor->stride, tmpStrides.data(), sizeof(long) * tmpStrides.size());
  }

  THCudaTensor_fill(state, tensor, 0.0f);
  return unique_ptr<THCudaTensor, CudaTensorDeleter>(
    tensor, CudaTensorDeleter(state));
}

unique_ptr<THCudaTensor, CudaTensorDeleter>
makeAliasedTHCudaTensorFull(THCState* state,
                            THCudaTensor* in,
                            const vector<long>& sizes,
                            const folly::Optional<vector<long>>& strides) {
  if (sizes.empty()) {
    throw invalid_argument("must have 1 or more dimensions");
  } else if (strides && strides->size() != sizes.size()) {
    throw invalid_argument("sizes and strides must match");
  }

  long size = 1;
  if (!strides) {
    for (const auto s : sizes) {
      size *= s;
    }
  } else {
    size = strides->front() * sizes.front();
  }

  long capacity = in->storage->size;

  if (capacity < size) {
    throw invalid_argument(
      "aliasing a THCudaTensor but new size overflows capacity");
  }

  auto sizesTH = (long*) THAlloc(sizeof(long) * sizes.size());
  SCOPE_EXIT { THFree(sizesTH); };
  memcpy(sizesTH, sizes.data(), sizeof(long) * sizes.size());

  auto stridesTH = (long*) THAlloc(sizeof(long) * sizes.size());
  SCOPE_EXIT { THFree(stridesTH); };
  if (strides) {
    memcpy(stridesTH, strides->data(), sizeof(long) * strides->size());
  } else {
    vector<long> tmpStrides(sizes.size());

    tmpStrides.back() = 1;
    for (int i = tmpStrides.size() - 2; i >= 0; --i) {
      tmpStrides[i] = tmpStrides[i + 1] * sizes[i + 1];
    }

    memcpy(stridesTH, tmpStrides.data(), sizeof(long) * tmpStrides.size());
  }

  auto szTH = LongStorage::wrap(
    makeMutable(LongRange(sizesTH, sizes.size()))).moveAsTH();
  SCOPE_EXIT { THLongStorage_free(szTH); };
  auto strTH = LongStorage::wrap(
    makeMutable(LongRange(stridesTH, sizes.size()))).moveAsTH();
  SCOPE_EXIT { THLongStorage_free(strTH); };

  auto tensor = THCudaTensor_newWithStorage(
    state,
    in->storage,
    in->storageOffset,
    szTH,
    strTH
  );

  return unique_ptr<THCudaTensor, CudaTensorDeleter>(
    tensor, CudaTensorDeleter(state));
}

std::unique_ptr<THCudaTensor, CudaTensorDeleter>
makeTHCudaTensorFull(THCState* state,
                     std::initializer_list<long> sizes,
                     std::initializer_list<long> strides) {
  if (strides.size()) {
    return makeTHCudaTensorFull(state,
                                vector<long>(sizes), vector<long>(strides));
  } else {
    return makeTHCudaTensorFull(state,
                                vector<long>(sizes));
  }
}

Tensor<float> copyFromCuda(THCState* state, const THCudaTensor* ctensor) {
  // Torch does not like constness but everything is safe here
  THCudaTensor* tensor = const_cast<THCudaTensor*>(ctensor);
  auto dataTH = Storage<float>(tensor->storage->size, 0.0f).moveAsTH();
  SCOPE_EXIT{ THFloatStorage_free(dataTH); };
  THFloatStorage_copyCuda(state, dataTH, tensor->storage);

  return Tensor<float>(
    Storage<float>(dataTH), tensor->storageOffset,
    LongStorage::wrap(
      makeMutable(LongRange(tensor->size, tensor->nDimension))),
    LongStorage::wrap(
      makeMutable(LongRange(tensor->stride, tensor->nDimension))));
}

unique_ptr<THCudaTensor, CudaTensorDeleter>
copyToCuda(THCState* state, Tensor<float>& tensor) {
  auto storageTH = tensor.storage().moveAsTH();
  SCOPE_EXIT{ THFloatStorage_free(storageTH); };

  auto sizeTH =
    LongStorage(tensor.sizes().begin(), tensor.sizes().end()).moveAsTH();
  SCOPE_EXIT{ THLongStorage_free(sizeTH); };

  auto strideTH =
    LongStorage(tensor.strides().begin(), tensor.strides().end()).moveAsTH();
  SCOPE_EXIT{ THLongStorage_free(strideTH); };

  auto cudaStorageTH =
    THCudaStorage_newWithSize(state, tensor.storage().size());
  SCOPE_EXIT{ THCudaStorage_free(state, cudaStorageTH); };
  THCudaStorage_copyFloat(state, cudaStorageTH, storageTH);

  return unique_ptr<THCudaTensor, CudaTensorDeleter>(
    THCudaTensor_newWithStorage(
      state, cudaStorageTH, tensor.storageOffset(), sizeTH, strideTH),
    CudaTensorDeleter(state));
}

} } }  // namespace

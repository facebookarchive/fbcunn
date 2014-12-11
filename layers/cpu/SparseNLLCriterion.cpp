// Copyright 2004-present Facebook. All Rights Reserved.
// Author: Michael Mathieu <myrhev@fb.com>

#include <algorithm>
#include <cstdio>
#include <memory>
#include <limits>

#include <lua.hpp>
#include <mkl.h>
#include <luaT.h>
#include <omp.h>

#include "Blas.h"
#include "torch/fb/fbcunn/layers/LuaUtils.h"
#include "torch/fb/fbcunn/layers/Tensor.h"
#include "Vml.h"
#include <folly/Format.h>

namespace facebook {
namespace deeplearning {
namespace torch {

namespace {

template <class T> using thOps = thpp::detail::TensorOps<T>;

template <class T>
int updateOutput(lua_State* L) {
  auto output    = luaGetFieldIfTensorChecked<T>(L, 1, "output");
  auto input     = luaGetTensorChecked<T>(L, 2);
  auto targetP   = luaGetTensorChecked<T>(L, 3);
  auto targetIdx = luaGetTensorChecked<long>(L, 4);
  auto batchSize = targetP.size(0);
  auto K = targetP.size(1);
  auto nClasses = input.size(1);
  luaL_argcheck(L, (output.ndims() == 1) && (output.size(0) == 1),
                1, "output has wrong dimension");
  luaL_argcheck(L, (input.ndims() == 2) && (input.size(0) == batchSize)
                && (input.isContiguous()),
                2, "input has wrong dimension");
  luaL_argcheck(L, (targetP.ndims() == 2) && (targetP.isContiguous()),
                3, "targetP has wrong dimension");
  luaL_argcheck(L, (targetIdx.ndims() == 2) && (targetIdx.size(0) == batchSize)
                && (targetIdx.size(1) == K) && (targetIdx.isContiguous()),
                3, "targetIdx has wrong dimension");

  auto targetIdxData = targetIdx.data();
  auto targetPData = targetP.data();
  auto inputData = input.data();

  T outputVal = 0.;
  for (int i = 0; i < batchSize; ++i) {
    auto targetIdxBatch = targetIdxData + i*K;
    auto targetPBatch = targetPData + i*K;
    auto inputBatch = inputData + i*nClasses;
    for (int j = 0; j < K; ++j) {
      outputVal += inputBatch[targetIdxBatch[j] - 1] * targetPBatch[j];
    }
  }

  output.data()[0] = - outputVal;

  return 0;
}

template <class T>
int updateGradInput(lua_State* L) {
  auto gradInput = luaGetFieldIfTensorChecked<T>(L, 1, "gradInput");
  auto targetP   = luaGetTensorChecked<T>(L, 2);
  auto targetIdx = luaGetTensorChecked<long>(L, 3);
  auto batchSize = targetP.size(0);
  auto K = targetP.size(1);
  auto nClasses = gradInput.size(1);
  luaL_argcheck(L, (gradInput.ndims() == 2) && (gradInput.size(0) == batchSize)
                && (gradInput.isContiguous()),
                1, "gradInput has wrong dimension");
  luaL_argcheck(L, (targetP.ndims() == 2) && (targetP.isContiguous()),
                2, "targetP has wrong dimension");
  luaL_argcheck(L, (targetIdx.ndims() == 2) && (targetIdx.size(0) == batchSize)
                && (targetIdx.size(1) == K) && (targetIdx.isContiguous()),
                2, "targetIdx has wrong dimension");

  auto targetIdxData = targetIdx.data();
  auto targetPData = targetP.data();
  auto gradInputData = gradInput.data();

  gradInput.zero();
  for (int i = 0; i < batchSize; ++i) {
    auto targetIdxBatch = targetIdxData + i*K;
    auto targetPBatch = targetPData + i*K;
    auto gradInputBatch = gradInputData + i*nClasses;
    for (int j = 0; j < K; ++j) {
      gradInputBatch[targetIdxBatch[j] - 1] = - targetPBatch[j];
    }
  }

  return 0;
}

template <class T>
class Registerer {
 private:
  static const luaL_Reg functions_[];
public:
  static void registerFunctions(lua_State* L);
};

template <class T>
const luaL_Reg Registerer<T>::functions_[] = {
  {"SparseNLLCriterion_updateOutput", updateOutput<T>},
  {"SparseNLLCriterion_updateGradInput", updateGradInput<T>},
  {nullptr, nullptr},
};

template <class T>
void Registerer<T>::registerFunctions(lua_State* L) {
  luaT_pushmetatable(L, Tensor<T>::kLuaTypeName);
  luaT_registeratname(L, functions_, "nn");
  lua_pop(L, 1);
}

} // namespace

void initSparseNLLCriterion(lua_State* L) {
  Registerer<float>::registerFunctions(L);
  Registerer<double>::registerFunctions(L);
}

}}} // namespaces

/**
 * Copyright 2014 Facebook
 * @author Tudor Bosman (tudorb@fb.com)
 */

#include <algorithm>
#include <cstdio>
#include <memory>

#include <lua.hpp>
#include <mkl.h>
#include <luaT.h>

#include "Blas.h"
#include "torch/fb/fbcunn/layers/LuaUtils.h"
#include "torch/fb/fbcunn/layers/Tensor.h"
#include "Vml.h"
#include "folly/Format.h"

namespace facebook { namespace deeplearning { namespace torch {

namespace {

// [batchSize][features][height][width]

// Compute output[i] = sum(fn(input[j])) for j s.t. |i - j| <= kernelSize
// It's equivalent to a 1d convolution (along the first dimension of input) of
// fn(input) with a kernelSize-sized kernel made up of all ones along the first
// dimension of input.
template <class T, class F>
void convolve(const Tensor<T>& input,
              Tensor<T>& output,
              int kernelSize,
              F fn) {
  int numChannels = input.size(0);
  long height = input.size(1);
  long width = input.size(2);

  long channelSize = height * width;

  const T* __restrict__ inptr = input.data();
  T* __restrict__ outptr = output.data();
  assert(inptr != outptr);

  const T* in = inptr;
  T* out = outptr;

  // Feature 0: sum for the first khalf features of input

  for (int pos = 0; pos < channelSize; ++pos) {
    out[pos] = fn(in[pos]);
  }
  in += channelSize;

  int khalf = kernelSize / 2;
  int kernelStride = kernelSize * channelSize;

  for (int ch = 1; ch < khalf + 1; ++ch) {
    for (int pos = 0; pos < channelSize; ++pos) {
      out[pos] += fn(in[pos]);
    }
    in += channelSize;
  }
  out += channelSize;

  // Features 1 .. khalf: at each step, we're adding one more feature from the
  // input

  for (int ch = khalf + 1; ch < kernelSize; ++ch) {
    for (int pos = 0; pos < channelSize; ++pos) {
      out[pos] = out[pos - channelSize] + fn(in[pos]);
    }
    in += channelSize;
    out += channelSize;
  }

  // Features khalf+1 .. end-khalf: at each step, we're adding one more
  // feature and subtracting an older feature

  for (int ch = kernelSize; ch < numChannels; ++ch) {
    for (int pos = 0; pos < channelSize; ++pos) {
      out[pos] = out[pos - channelSize] - fn(in[pos - kernelStride]) +
        fn(in[pos]);
    }
    in += channelSize;
    out += channelSize;
  }

  // Features end-khalf+1 .. end-1: at each step, we're subtracting an older
  // feature

  for (int ch = numChannels; ch < numChannels + khalf; ++ch) {
    for (int pos = 0; pos < channelSize; ++pos) {
      out[pos] = out[pos - channelSize] - fn(in[pos - kernelStride]);
    }
    out += channelSize;
    in += channelSize;
  }
}

// compute the denominator (before raising it to power)
//
// for each pixel,
// out[i][y][x] = 1 + scale * sum(in[j][x][y] ** 2)
//                            for j, |j - i| <= kernelSize
template <class T>
void computeDenominator(
    const Tensor<T>& input,
    Tensor<T>& output,
    int kernelSize,
    T scale) {
  convolve(input, output, kernelSize, [] (T x) { return x * x; });

  long n = input.size();
  T* outptr = output.data();
  for (int i = 0; i < n; ++i) {
    outptr[i] = 1 + scale * outptr[i];
  }
}

// Forward pass (one image)
template <class T>
void updateOutputForImage(
    const Tensor<T>& input,
    Tensor<T>& output,
    int kernelSize,
    T scale,
    T power) {
  computeDenominator(input, output, kernelSize, scale);

  long n = input.size();
  const T* inptr = input.data();
  T* outptr = output.data();

  vml::powx(n, outptr, -power, outptr);
  vml::mul(n, inptr, outptr, outptr);
}

// Forward pass
template <class T>
int updateOutput(lua_State* L) {
  auto input = luaGetTensorChecked<T>(L, 2);
  auto output = luaGetFieldIfTensorChecked<T>(L, 1, "output");
  int outputIdx = lua_gettop(L);

  int kernelSize = luaGetFieldIfNumberChecked<int>(L, 1, "size");
  T scale = luaGetFieldIfNumberChecked<T>(L, 1, "scale");
  T power = luaGetFieldIfNumberChecked<T>(L, 1, "power");

  if (kernelSize % 2 == 0) {
    luaL_error(L, "Kernel size must be odd");
  }

  scale /= kernelSize;

  int ndims = input.ndims();
  if (ndims != 3 && ndims != 4) {
    luaL_error(L, "Invalid input tensor dimension");
  }

  output.resizeAs(input);

  if (ndims == 4) {
    // batched: the first dimension is the batch size
    long batchSize = input.size(0);
    Tensor<T> input1;
    Tensor<T> output1;

    // TODO(#3821228): OpenMP
    for (long imageIdx = 0; imageIdx < batchSize; ++imageIdx) {
      input1.select(input, 0, imageIdx);
      output1.select(output, 0, imageIdx);
      updateOutputForImage(input1, output1, kernelSize, scale, power);
    }
  } else {
    updateOutputForImage(input, output, kernelSize, scale, power);
  }

  lua_pushvalue(L, outputIdx);
  return 1;
}

// Get tempCount temporary tensors.
//
// They're stored under the name _tmp in the Lua object at index selfIdx
// on the stack, and created the first time through this function.
template <class T>
std::vector<Tensor<T>> getTempTensors(lua_State* L, int selfIdx,
                                      int tempCount,
                                      LongRange sizes) {
  lua_getfield(L, selfIdx, "_tmp");     // _tmp
  if (lua_isnil(L, -1)) {
    lua_pop(L, 1);                      // <empty>
    lua_createtable(L, tempCount, 0);   // new_table
    lua_pushvalue(L, -1);               // new_table new_table
    lua_setfield(L, 1, "_tmp");         // new_table
  } else {
    luaL_checktype(L, -1, LUA_TTABLE);
  }

  int tableIdx = lua_gettop(L);

  std::vector<Tensor<T>> tempTensors;
  tempTensors.reserve(tempCount);

  for (int i = 1; i <= tempCount; ++i) {
    Tensor<T> tensor {TensorInvalid()};
    lua_rawgeti(L, tableIdx, i);
    if (lua_isnil(L, -1)) {
      lua_pop(L, 1);
      tensor = Tensor<T>();
      luaPushTensor(L, tensor);
      lua_rawseti(L, tableIdx, i);
    } else {
      tensor = luaGetTensorChecked<T>(L, -1);
      lua_pop(L, 1);
    }
    tensor.resize(sizes);
    tempTensors.push_back(std::move(tensor));
  }

  return tempTensors;
}

// Backprop (one image)
template <class T>
void updateGradInputForImage(
    const Tensor<T>& inputTensor,
    const Tensor<T>& gradOutputTensor,
    Tensor<T>& gradInputTensor,
    int kernelSize,
    T scale,
    T power,
    std::vector<Tensor<T>>& tmpTensors) {
  const T* input = inputTensor.data();
  const T* gradOutput = gradOutputTensor.data();
  T* gradInput = gradInputTensor.data();
  long n = inputTensor.size();

  // d[j] = 1 + scale * sum(x[k] ** 2), for k s.t |k-j| <= kernelSize
  auto& dTensor = tmpTensors[0];
  auto d = dTensor.data();
  computeDenominator(inputTensor, dTensor, kernelSize, scale);

  // we need x[j] * d[j] ** (-power - 1) and d[j] ** (-power)
  //
  // compute s = d ** (-power)
  //         d = s / d * x

  auto& sTensor = tmpTensors[1];
  auto s = sTensor.data();
  vml::powx(n, d, -power, s);
  vml::mul(n, gradOutput, s, s);

  vml::div(n, s, d, d);
  vml::mul(n, input, d, d);
  blas::scal(n, -2 * scale * power, d, 1);

  // gi[j] = sum(d[k]) for k s.t |k-j| <= kernelSize
  convolve(dTensor, gradInputTensor, kernelSize, [] (T v) { return v; });

  // gi[j] = x[j] * gi[j] + s[j]
  vml::mul(n, input, gradInput, gradInput);
  vml::add(n, s, gradInput, gradInput);
}

// Backprop
template <class T>
int updateGradInput(lua_State* L) {
  auto input = luaGetTensorChecked<T>(L, 2);
  auto gradOutput = luaGetTensorChecked<T>(L, 3);
  auto gradInput = luaGetFieldIfTensorChecked<T>(L, 1, "gradInput");
  int gradInputIdx = lua_gettop(L);

  int kernelSize = luaGetFieldIfNumberChecked<int>(L, 1, "size");
  T scale = luaGetFieldIfNumberChecked<T>(L, 1, "scale");
  T power = luaGetFieldIfNumberChecked<T>(L, 1, "power");

  if (kernelSize % 2 == 0) {
    luaL_error(L, "Kernel size must be odd");
  }

  scale /= kernelSize;

  int ndims = input.ndims();
  if (ndims != 3 && ndims != 4) {
    luaL_error(L, "Invalid input tensor dimension");
  }

  gradInput.resizeAs(input);

  LongRange tmpTensorDim = input.sizes();
  if (ndims == 4) {
    tmpTensorDim.pop_front();  // first dim is batch size
  }
  auto tmpTensors = getTempTensors<T>(L, 1, 2, tmpTensorDim);

  if (ndims == 4) {
    long batchSize = input.size(0);
    Tensor<T> input1;
    Tensor<T> gradOutput1;
    Tensor<T> gradInput1;

    for (long imageIdx = 0; imageIdx < batchSize; ++imageIdx) {
      input1.select(input, 0, imageIdx);
      gradOutput1.select(gradOutput, 0, imageIdx);
      gradInput1.select(gradInput, 0, imageIdx);
      updateGradInputForImage(input1, gradOutput1, gradInput1,
                              kernelSize, scale, power, tmpTensors);
    }
  } else {
    updateGradInputForImage(input, gradOutput, gradInput, kernelSize, scale,
                            power, tmpTensors);
  }

  lua_pushvalue(L, gradInputIdx);
  return 1;
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
  {"CrossMapNormalization_updateOutput", updateOutput<T>},
  {"CrossMapNormalization_updateGradInput", updateGradInput<T>},
  {nullptr, nullptr},
};

template <class T>
void Registerer<T>::registerFunctions(lua_State* L) {
  luaT_pushmetatable(L, Tensor<T>::kLuaTypeName);
  luaT_registeratname(L, functions_, "nn");
  lua_pop(L, 1);
}

}  // namespace

void initCrossMapNormalization(lua_State* L) {
  Registerer<float>::registerFunctions(L);
  Registerer<double>::registerFunctions(L);
}

}}}  // namespaces

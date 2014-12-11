/**
 * Copyright 2014 Facebook
 * @author Frank Jargstorff (fjargsto@fb.com)
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
#include <folly/Format.h>
#include <cassert>

namespace facebook { namespace deeplearning { namespace torch {

namespace {

template <class T>
void updateOutput1(const Tensor<T> & input, const Tensor<T> & weight,
                   const Tensor<T> & bias, Tensor<T> & output,
                   int dW, int dH) {
  const int oP = weight.size(0);
  const int oH = weight.size(1);
  const int oW = weight.size(2);

  const int iP = weight.size(3);
  const int kH = weight.size(4);
  const int kW = weight.size(5);

  for (auto oPlane = 0; oPlane < oP; ++oPlane) {
    for (auto oRow = 0; oRow < oH; ++oRow) {
      for (auto oCol = 0; oCol < oW; ++oCol) {
        T sum = bias.at({oPlane});
        for (auto iPlane = 0; iPlane < iP; ++iPlane) {
          for (auto kRow = 0; kRow < kH; ++kRow) {
            for (auto kCol = 0; kCol < kW; ++kCol) {
              sum += input.at({iPlane, oRow*dH + kRow, oCol*dW + kCol})
                   * weight.at({oPlane, oRow, oCol, iPlane, kRow, kCol});
            }
          }
        }
        output.at({oPlane, oRow, oCol}) = sum;
      }
    }
  }
}

// Forward pass
template <class T>
int updateOutput(lua_State* L) {
  auto input = luaGetTensorChecked<T>(L, 2);
  auto output = luaGetFieldIfTensorChecked<T>(L, 1, "output");
  // store output index
  int outputIdx = lua_gettop(L);

  auto weight = luaGetFieldIfTensorChecked<T>(L, 1, "weight");
  auto bias   = luaGetFieldIfTensorChecked<T>(L, 1, "bias");

  int dW = luaGetFieldIfNumberChecked<int>(L, 1, "dW");
  int dH = luaGetFieldIfNumberChecked<int>(L, 1, "dH");

  if (input.ndims() == 4) {
    // batched: the first dimension is the batch size
    long batchSize = input.size(0);
    Tensor<T> input1;
    Tensor<T> output1;

    for (long batchIdx = 0; batchIdx < batchSize; ++batchIdx) {
      input1.select(input, 0, batchIdx);
      output1.select(output, 0, batchIdx);
      updateOutput1(input1, weight, bias, output1, dW, dH);
    }
  } else {
    // resize output tensor
    updateOutput1(input, weight, bias, output, dW, dH);
  }

  // push copy of stack location 'outputIdx' onto the stack
  lua_pushvalue(L, outputIdx);
  return 1;
}

template <class T>
void updateGradInput1(const Tensor<T> & gradOutput, const Tensor<T> & weight,
                      Tensor<T> & gradInput, int dW, int dH) {
  const int iP = gradInput.size(0);
  const int iH = gradInput.size(1);
  const int iW = gradInput.size(2);

  const int oP = weight.size(0);
  const int oH = weight.size(1);
  const int oW = weight.size(2);
  const int kH = weight.size(4);
  const int kW = weight.size(5);

  for (auto iPlane = 0; iPlane < iP; ++iPlane) {
    for (auto iRow = 0; iRow < iH; ++iRow) {
      for (auto iCol = 0; iCol < iW; ++iCol) {
        T sum = 0;
        for (auto oPlane = 0; oPlane < oP; ++oPlane) {
          for (auto oRow = std::max(0, (iRow - kH)/dH);
               oRow < std::min(iRow/dH+1, oH); ++oRow) {
            for (auto oCol = std::max(0, (iCol - kW)/dW);
                 oCol < std::min(iCol/dW+1, oW); ++oCol) {
              auto kRow = iRow - dH * oRow;
              auto kCol = iCol - dW * oCol;
              // this guard basically implements zero padding for the
              // "convolution" performed. nothing more than guards
              // to not overrun in source arrays (weight and gradOutput).
              if (0 <= kRow && kRow < kH && 0 <= kCol && kCol < kW) {
                sum += gradOutput.at({oPlane, oRow, oCol})
                    * weight.at({oPlane, oRow, oCol,
                                 iPlane, kRow, kCol});
              }
            }
          }
        }
        gradInput.at({iPlane, iRow, iCol}) = sum;
      }
    }
  }
}

// Backprop
template <class T>
int updateGradInput(lua_State* L) {
  auto input = luaGetTensorChecked<T>(L, 2);
  auto gradOutput = luaGetTensorChecked<T>(L, 3);
  auto gradInput  = luaGetFieldIfTensorChecked<T>(L, 1, "gradInput");
  // store output index
  int gradInputIdx = lua_gettop(L);

  auto weight = luaGetFieldIfTensorChecked<T>(L, 1, "weight");

  int dW = luaGetFieldIfNumberChecked<int>(L, 1, "dW");
  int dH = luaGetFieldIfNumberChecked<int>(L, 1, "dH");

  gradInput.resizeAs(input);
  if (input.ndims() == 4) { // batch
    // batched: the first dimension is the batch size
    long batchSize = input.size(0);
    Tensor<T> gradInput1;
    Tensor<T> gradOutput1;

    for (long batchIdx = 0; batchIdx < batchSize; ++batchIdx) {
      gradOutput1.select(gradOutput, 0, batchIdx);
      gradInput1.select(gradInput, 0, batchIdx);
      updateGradInput1(gradOutput1, weight, gradInput1, dW, dH);
    }
  } else { // non-batched
    updateGradInput1(gradOutput, weight, gradInput, dW, dH);
  }

  lua_pushvalue(L, gradInputIdx);
  return 1;
}

template <class T>
void accGradParameters1(const Tensor<T> & input, const Tensor<T> & gradOutput,
                        T scale, Tensor<T> & gradWeight, Tensor<T> & gradBias,
                        int dW, int dH) {
  const int iP = input.size(0);
  const int iH = input.size(1);
  const int iW = input.size(2);

  const int oP = gradWeight.size(0);
  const int oH = gradWeight.size(1);
  const int oW = gradWeight.size(2);
  const int kH = gradWeight.size(4);
  const int kW = gradWeight.size(5);

  for (auto oPlane = 0; oPlane < oP; ++oPlane) {
    T bias = 0.0;
    for (auto oRow = 0; oRow < oH; ++oRow) {
      for (auto oCol = 0; oCol < oW; ++oCol) {
        bias += scale * gradOutput.at({oPlane, oRow, oCol});
        for (auto iPlane = 0; iPlane < iP; ++iPlane) {
          for (auto kRow = 0; kRow < kH; ++kRow) {
            for (auto kCol = 0; kCol < kW; ++kCol) {
              int iRow = dH * oRow + kRow;
              int iCol = dW * oCol + kCol;
              gradWeight.at({oPlane, oRow, oCol, iPlane, kRow, kCol}) +=
                  scale * gradOutput.at({oPlane, oRow, oCol}) *
                  input.at({iPlane, iRow, iCol});
            }
          }
        }
      }
    }
    gradBias.at({oPlane}) += bias;
  }
}

// Weight update/gradient
template <class T>
int accGradParameters(lua_State* L) {
  auto input = luaGetTensorChecked<T>(L, 2);
  auto gradOutput = luaGetTensorChecked<T>(L, 3);

  T scale = luaGetNumberChecked<T>(L, 4);

  auto gradWeight = luaGetFieldIfTensorChecked<T>(L, 1, "gradWeight");
  auto gradBias   = luaGetFieldIfTensorChecked<T>(L, 1, "gradBias");

  int dW = luaGetFieldIfNumberChecked<int>(L, 1, "dW");
  int dH = luaGetFieldIfNumberChecked<int>(L, 1, "dH");

  if (input.ndims() == 4) { // batch
    // batched: the first dimension is the batch size
    long batchSize = input.size(0);
    Tensor<T> input1;
    Tensor<T> gradOutput1;

    for (long batchIdx = 0; batchIdx < batchSize; ++batchIdx) {
      input1.select(input, 0, batchIdx);
      gradOutput1.select(gradOutput, 0, batchIdx);
      accGradParameters1(input1, gradOutput1, scale, gradWeight, gradBias,
                         dW, dH);
    }
  } else { // non-batched
    accGradParameters1(input, gradOutput, scale, gradWeight, gradBias,
                       dW, dH);
  }
  return 0;
}

template <class T>
class Registrar {
 private:
  static const luaL_Reg functions_[];

 public:
  static void registerFunctions(lua_State* L);
};

template <class T>
const luaL_Reg Registrar<T>::functions_[] = {
  {"LocallyConnected_updateOutput", updateOutput<T>},
  {"LocallyConnected_updateGradInput", updateGradInput<T>},
  {"LocallyConnected_accGradParameters", accGradParameters<T>},
  {nullptr, nullptr},
};

template <class T>
void Registrar<T>::registerFunctions(lua_State* L) {
  luaT_pushmetatable(L, Tensor<T>::kLuaTypeName);
  luaT_registeratname(L, functions_, "nn");
  lua_pop(L, 1);
}

}  // anonymous namespace

void initLocallyConnected(lua_State* L) {
  Registrar<float>::registerFunctions(L);
  Registrar<double>::registerFunctions(L);
}

}}}  // facebook::deeplearning::torch namespace

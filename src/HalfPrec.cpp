// Copyright 2004-, Facebook, Inc. All Rights Reserved.

#include "HalfPrec.h"

#include <string>
#include <assert.h>
#include <lua.hpp>
#include "Utils.h"
#include "Tensor.h"
#include "LuaUtils.h"
#include "THC.h"

using namespace std;
using namespace facebook::deeplearning::torch;

namespace {

// Would be nice to use thrust, but its header files are full of things
// that trip our Werr settings for unused typedefs.
struct HalfTensor {
  explicit HalfTensor(THCState* state, THCudaTensor* floats) {
    auto sz = THCudaTensor_nElement(state, floats);
    auto err = cudaMalloc(&devPtr_, sz * sizeof(half_t));
    if (err != cudaSuccess) {
      throw std::runtime_error("failed to cudamalloc HalfTensor");
    }
    size_ = sz;
    halfprec_ToHalf(THCState_getCurrentStream(state),
                    THCudaTensor_data(state, floats), devPtr_, size_);
  }

  HalfTensor()
  : devPtr_(nullptr)
  , size_(0) {
  }

  ~HalfTensor() {
    cudaFree(devPtr_);
  }

  void toFloat(THCState* state, THCudaTensor* dest) {
    THCudaTensor_resize1d(state, dest, size_);
    assert(size_ > 0);
    halfprec_ToFloat(THCState_getCurrentStream(state),
                     devPtr_, THCudaTensor_data(state, dest), size_);
  }

private:
  half_t *devPtr_;
  size_t size_;
};

const char* kLibName = "HalfPrec";

int HalfPrec_new(lua_State* l) {
  auto dv = new HalfTensor();
  luaT_pushudata(l, dv, kLibName);
  return 1;
}

int HalfPrec_destroy(lua_State* l) {
  delete static_cast<HalfTensor*>(luaT_checkudata(l, 1, kLibName));
  return 0;
};

int HalfPrec_toHalfCUDA(lua_State* l) {
  THCState* state = getCutorchState(l);
  auto input = (THCudaTensor*)luaT_checkudata(l, 1, "torch.CudaTensor");
  THAssert(THCudaTensor_checkGPU(state, 1, input));
  auto cinput = THCudaTensor_newContiguous(state, input);
  auto dest = new HalfTensor(state, cinput);

  luaT_pushudata(l, dest, kLibName);
  THCudaTensor_free(state, cinput);
  return 1;
}

int HalfPrec_toFloatCUDA(lua_State* l) {
  THCState* state = getCutorchState(l);
  auto input = (HalfTensor*)luaT_checkudata(l, 1, kLibName);
  auto dest = THCudaTensor_new(state);
  input->toFloat(state, dest);
  luaT_pushudata(l, dest, "torch.CudaTensor");
  return 1;
}

const struct luaL_reg manifest[] = {
  {"new", HalfPrec_new},
  {"toHalfCUDA", HalfPrec_toHalfCUDA},
  {"toFloatCUDA", HalfPrec_toFloatCUDA},
  {"free", HalfPrec_destroy},
  {nullptr, nullptr},
};

}

extern "C" int luaopen_libhalfprec(lua_State* L) {
  luaT_newmetatable(L, kLibName, nullptr,
                    HalfPrec_new, // ctor
                    HalfPrec_destroy, // dtor
                    nullptr);
  lua_newtable(L);
  luaL_register(L, nullptr, manifest);
  return 1;
}

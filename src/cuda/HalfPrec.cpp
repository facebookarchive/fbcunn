// Copyright 2004-, Facebook, Inc. All Rights Reserved.

#include "HalfPrec.h"

#include <string>
#include <assert.h>
#include <lua.hpp>
#include "torch/fb/fbcunn/layers/Tensor.h"
#include "torch/fb/fbcunn/layers/LuaUtils.h"
#include "THC.h"

using namespace std;

namespace {

// Would be nice to use thrust, but its header files are full of things
// that trip our Werr settings for unused typedefs.
struct HalfTensor {
  explicit HalfTensor(THCudaTensor* floats) {
    auto sz = THCudaTensor_nElement(NULL, floats);
    auto err = cudaMalloc(&devPtr_, sz * sizeof(half_t));
    if (err != cudaSuccess) {
      throw std::runtime_error("failed to cudamalloc HalfTensor");
    }
    size_ = sz;
    halfprec_ToHalf(THCudaTensor_data(NULL, floats), devPtr_, size_);
  }

  HalfTensor()
  : devPtr_(nullptr)
  , size_(0) {
  }

  ~HalfTensor() {
    cudaFree(devPtr_);
  }

  void toFloat(THCudaTensor* dest) {
    THCudaTensor_resize1d(NULL, dest, size_);
    assert(size_ > 0);
    halfprec_ToFloat(devPtr_, THCudaTensor_data(NULL, dest), size_);
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
  auto input = (THCudaTensor*)luaT_checkudata(l, 1, "torch.CudaTensor");
  auto cinput = THCudaTensor_newContiguous(NULL, input);
  auto dest = new HalfTensor(cinput);

  luaT_pushudata(l, dest, kLibName);
  THCudaTensor_free(NULL, cinput);
  return 1;
}

int HalfPrec_toFloatCUDA(lua_State* l) {
  auto input = (HalfTensor*)luaT_checkudata(l, 1, kLibName);
  auto dest = THCudaTensor_new(NULL);
  input->toFloat(dest);
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

/**
 * Copyright 2015 Facebook
 */

#include "cuda/DeviceTensor.cuh"
#include "src/Utils.h"
#include "src/DeviceTensorUtils.h"
#include "THC.h"

#include <lua.hpp>
#include <TH.h>
#include <luaT.h>

using namespace facebook::cuda;

namespace facebook { namespace deeplearning { namespace torch {

namespace detail {
void launchWeightedLookupTableScaleByWeightKernel(
  cudaStream_t stream,
  DeviceTensor<float, 2>& output,
  DeviceTensor<float, 2>& input,
  DeviceTensor<float, 1>& weight);
}

namespace {

int scaleByWeight(lua_State* L) {
  THCState* state = getCutorchState(L);
  auto output  = (THCudaTensor*)luaT_checkudata(L, 1, "torch.CudaTensor");
  const auto input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  const auto weight = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");

  DeviceTensor<float, 2> cudaOutput = torchToDeviceTensor<float, 2>(state, output);
  DeviceTensor<float, 2> cudaInput = torchToDeviceTensor<float, 2>(state, input);
  DeviceTensor<float, 1> cudaWeight = torchToDeviceTensor<float, 1>(state, weight);

  detail::launchWeightedLookupTableScaleByWeightKernel(
    THCState_getCurrentStream(state),
    cudaOutput, cudaInput, cudaWeight);

  return 0;
}

const luaL_Reg functions[] = {
  {"WeightedLookupTable_scaleByWeight", scaleByWeight},
  {nullptr, nullptr},
};

} // namespace

void initWeightedLookupTableCuda(lua_State* L) {
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, functions, "nn");
  lua_pop(L, 1);
}

}}}  // namespaces

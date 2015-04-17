// Copyright 2014 Facebook

#include "Utils.h"
#include "../Utils.h"
#include "CudaTensorUtils.h"
#include "CuFFTStrategy.h"
#include "SpatialConvolutionCuFFT.h"
#include "SpatialConvolutionCuFFTTuner.h"

#include <luaT.h>
#include <lua.hpp>
#include <vector>

using namespace std;

namespace facebook { namespace deeplearning { namespace torch {

namespace {
// Forward pass, converts to 4-D with batch size 1 if needed
int updateOutputLua(lua_State* L) {
  THCState* state = getCutorchState(L);
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  auto weight =
    (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "weight", "torch.CudaTensor");
  auto bias =
    (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "bias", "torch.CudaTensor");
  auto output =
    (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");

  auto input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");

  luaL_argcheck(L, THCudaTensor_nDimension(state, input) == 4, 2,
                "4D (batch mode) tensor is expected");
  luaL_argcheck(L, THCudaTensor_isContiguous(state, input), 2,
                "input must be contiguous");
  luaL_argcheck(L, THCudaTensor_isContiguous(state, weight), 1,
                "weight must be contiguous");
  luaL_argcheck(L, THCudaTensor_isContiguous(state, bias), 1,
                "bias must be contiguous");

  luaL_argcheck(L,
                THCudaTensor_size(state, input, 1) ==
                THCudaTensor_size(state, weight, 1),
                2,
                "Wrong number of input planes");
  luaL_argcheck(L, (dH == 1) && (dW == 1), 1, "FFT only supports stride 1");

  THBuffers bufs;
  bufs.input = (THCudaTensor*)luaT_getfieldcheckudata(
    L, 1, "inputBuffer", "torch.CudaTensor");
  bufs.inputTranspose = (THCudaTensor*)luaT_getfieldcheckudata(
    L, 1, "inputTransposeBuffer", "torch.CudaTensor");
  bufs.output = (THCudaTensor*)luaT_getfieldcheckudata(
    L, 1, "outputBuffer", "torch.CudaTensor");
  bufs.outputTranspose = (THCudaTensor*)luaT_getfieldcheckudata(
    L, 1, "outputTransposeBuffer", "torch.CudaTensor");
  bufs.weight = (THCudaTensor*)luaT_getfieldcheckudata(
    L, 1, "weightBuffer", "torch.CudaTensor");
  bufs.weightTranspose = (THCudaTensor*)luaT_getfieldcheckudata(
    L, 1, "weightTransposeBuffer", "torch.CudaTensor");

  THAssert(THCudaTensor_checkGPU(state, 10, weight, bias, output, input,
                                 bufs.input, bufs.inputTranspose,
                                 bufs.output, bufs.outputTranspose,
                                 bufs.weight, bufs.weightTranspose));

  THParams thp(state, input, weight, output, bias, 0.0f, bufs);
  ConvolutionPass pass(ConvolutionPass(ConvolutionPass::kUpdateOutput));
  ProblemSizes pbs(thp, pass);

  auto strategy = SpatialConvolutionCuFFTTuner::getBestPerformance(state, pbs);
  if (!strategy) {
    luaL_error(L, "FFT problem too large; no viable strategy found");
  }

  detail::updateOutputTH(state, thp, pbs, *strategy);
  return 0;
}

// Backprop, converts to 4-D with batch size 1 if needed
int updateGradInputLua(lua_State* L) {
  THCState* state = getCutorchState(L);
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  auto weight =
    (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "weight", "torch.CudaTensor");
  auto gradInput =
    (THCudaTensor*)luaT_getfieldcheckudata(
      L, 1, "gradInput", "torch.CudaTensor");
  auto gradOutput = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");

  luaL_argcheck(L, THCudaTensor_nDimension(state, gradOutput) == 4, 2,
                "4D (batch mode) tensor is expected");
  luaL_argcheck(L, THCudaTensor_isContiguous(state, gradOutput), 2,
                "gradOutput must be contiguous");
  luaL_argcheck(L, THCudaTensor_isContiguous(state, weight), 1,
                "weight must be contiguous");

  luaL_argcheck(L, (dH == 1) && (dW == 1), 1, "FFT only supports stride 1");

  THBuffers bufs;
  bufs.input = (THCudaTensor*)luaT_getfieldcheckudata(
    L, 1, "inputBuffer", "torch.CudaTensor");
  bufs.inputTranspose = (THCudaTensor*)luaT_getfieldcheckudata(
    L, 1, "inputTransposeBuffer", "torch.CudaTensor");
  bufs.output = (THCudaTensor*)luaT_getfieldcheckudata(
    L, 1, "outputBuffer", "torch.CudaTensor");
  bufs.outputTranspose = (THCudaTensor*)luaT_getfieldcheckudata(
    L, 1, "outputTransposeBuffer", "torch.CudaTensor");
  bufs.weight = (THCudaTensor*)luaT_getfieldcheckudata(
    L, 1, "weightBuffer", "torch.CudaTensor");
  bufs.weightTranspose = (THCudaTensor*)luaT_getfieldcheckudata(
    L, 1, "weightTransposeBuffer", "torch.CudaTensor");

  THAssert(THCudaTensor_checkGPU(state, 9, weight, gradInput, gradOutput,
                                 bufs.input, bufs.inputTranspose,
                                 bufs.output, bufs.outputTranspose,
                                 bufs.weight, bufs.weightTranspose));

  THParams thp(state, gradInput, weight, gradOutput, nullptr, 0.0f, bufs);
  ConvolutionPass pass(ConvolutionPass(ConvolutionPass::kUpdateGradInput));
  ProblemSizes pbs(thp, pass);

  auto strategy = SpatialConvolutionCuFFTTuner::getBestPerformance(state, pbs);
  if (!strategy) {
    luaL_error(L, "FFT problem too large; no viable strategy found");
  }

  detail::updateGradInputTH(state, thp, pbs, *strategy);
  return 0;
}

// Backprop, converts to 4-D with batch size 1 if needed
int accGradParametersLua(lua_State* L) {
  THCState* state = getCutorchState(L);
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  auto gradWeight =
    (THCudaTensor*)luaT_getfieldcheckudata(
      L, 1, "gradWeight", "torch.CudaTensor");
  auto gradBias =
    (THCudaTensor*)luaT_getfieldcheckudata(
      L, 1, "gradBias", "torch.CudaTensor");
  auto input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  auto gradOutput = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
  float scale = luaL_optnumber(L, 4, 1.f);

  luaL_argcheck(L, THCudaTensor_nDimension(state, input) == 4, 2,
                "4D (batch mode) tensor is expected");
  luaL_argcheck(L, THCudaTensor_isContiguous(state, input), 2,
                "input must be contiguous");
  luaL_argcheck(L, THCudaTensor_isContiguous(state, gradOutput), 3,
                "gradOutput must be contiguous");
  luaL_argcheck(L, THCudaTensor_isContiguous(state, gradBias), 1,
                "gradBias must be contiguous");

  luaL_argcheck(L, (dH == 1) && (dW == 1), 1, "FFT only supports stride 1");

  THBuffers bufs;
  bufs.input = (THCudaTensor*)luaT_getfieldcheckudata(
    L, 1, "inputBuffer", "torch.CudaTensor");
  bufs.inputTranspose = (THCudaTensor*)luaT_getfieldcheckudata(
    L, 1, "inputTransposeBuffer", "torch.CudaTensor");
  bufs.output = (THCudaTensor*)luaT_getfieldcheckudata(
    L, 1, "outputBuffer", "torch.CudaTensor");
  bufs.outputTranspose = (THCudaTensor*)luaT_getfieldcheckudata(
    L, 1, "outputTransposeBuffer", "torch.CudaTensor");
  bufs.weight = (THCudaTensor*)luaT_getfieldcheckudata(
    L, 1, "weightBuffer", "torch.CudaTensor");
  bufs.weightTranspose = (THCudaTensor*)luaT_getfieldcheckudata(
    L, 1, "weightTransposeBuffer", "torch.CudaTensor");

  THAssert(THCudaTensor_checkGPU(state, 10, gradWeight, gradBias, input,
                                 gradOutput, bufs.input, bufs.inputTranspose,
                                 bufs.output, bufs.outputTranspose,
                                 bufs.weight, bufs.weightTranspose));

  THParams thp(state, input, gradWeight, gradOutput, gradBias, scale, bufs);
  ConvolutionPass pass(ConvolutionPass(ConvolutionPass::kAccGradParameters));
  ProblemSizes pbs(thp, pass);

  auto strategy = SpatialConvolutionCuFFTTuner::getBestPerformance(state, pbs);
  if (!strategy) {
    luaL_error(L, "FFT problem too large; no viable strategy found");
  }

  detail::accGradParametersTH(state, thp, pbs, *strategy);
  return 0;
}

int cleanupBuffersLua(lua_State* L) {
  detail::cleanupBuffers();
  return 0;
}

namespace {

vector<long> defaultBatches({128});
vector<long> defaultFilters({128});
vector<long> defaultPlanes({128});
vector<long> defaultInputRows({16});
vector<long> defaultInputCols({16});
vector<long> defaultWeightRows({3});
// If weightCols are unspecified, use only square weights
vector<long> defaultWeightCols({});

vector<long> asVector(
  THCState* state, THCudaTensor* th, const vector<long>& defaultVec) {
  if (THCudaTensor_nDimension(state, th) == 0) {
    return defaultVec;
  }
  auto t = copyFromCuda(state, th);
  CHECK_EQ(1, t.ndims());
  return vector<long>(t.data(), t.data() + t.size());
}

}

// Given THCudaTensor containing sizes of interest and passed above from
// lua, does a search over the cartesian product of all sizes.
// Comparison across sizes can be made using the GReductions per second
// metric.
int explorePerformanceLua(lua_State* L) {
  THCState* state = getCutorchState(L);
  auto batchesTensor =
    (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  auto filtersTensor =
    (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
  auto planesTensor =
    (THCudaTensor*)luaT_checkudata(L, 4, "torch.CudaTensor");
  auto inputRowsTensor =
    (THCudaTensor*)luaT_checkudata(L, 5, "torch.CudaTensor");
  auto inputColsTensor =
    (THCudaTensor*)luaT_checkudata(L, 6, "torch.CudaTensor");
  auto weightRowsTensor =
    (THCudaTensor*)luaT_checkudata(L, 7, "torch.CudaTensor");
  auto weightColsTensor =
    (THCudaTensor*)luaT_checkudata(L, 8, "torch.CudaTensor");

  THAssert(THCudaTensor_checkGPU(state, 7, batchesTensor, filtersTensor,
                                 planesTensor, inputRowsTensor, inputColsTensor,
                                 weightRowsTensor, weightColsTensor));

  auto batches = asVector(state, batchesTensor, defaultBatches);
  auto filters = asVector(state, filtersTensor, defaultFilters);
  auto planes = asVector(state, planesTensor, defaultPlanes);
  auto inputRows = asVector(state, inputRowsTensor, defaultInputRows);
  auto inputCols = asVector(state, inputColsTensor, defaultInputCols);
  auto weightRows = asVector(state, weightRowsTensor, defaultWeightRows);
  auto weightCols = asVector(state, weightColsTensor, defaultWeightCols);

  for (auto batch : batches) {
    for (auto filter : filters) {
      for (auto plane : planes) {
        for (auto inputSizeRow : inputRows) {
          for (auto inputSizeCol : inputCols) {
            // Keep problem size manageable
            CHECK_LT(0, inputSizeRow);
            CHECK_LT(0, inputSizeCol);
            ProblemSizes pbs = ProblemSizes().withBatch(batch).
              withFilter(filter).
              withPlane(plane).
              withInputSizeRow(inputSizeRow).
              withInputSizeCol(inputSizeCol).
              withOutputSizeRow(inputSizeRow).
              withOutputSizeCol(inputSizeCol);

            for (auto weightSizeRow : weightRows) {
              // If weightCols are unspecified, use only square weights
              auto weightColsLocal = (weightCols.size() == 0) ?
                vector<long>({weightSizeRow}) : weightCols;
              for (auto weightSizeCol : weightColsLocal) {
                if (weightSizeRow > inputSizeRow ||
                    weightSizeCol > inputSizeCol) {
                  continue;
                }

                pbs.withWeightSizeRow(weightSizeRow).
                  withWeightSizeCol(weightSizeCol);

                SpatialConvolutionCuFFTTuner::
                  getBestPerformance(state, pbs);
              }
            }
          }
        }
      }
    }
  }
  return 0;
}

const luaL_Reg functions[] = {
  {"SpatialConvolutionCuFFT_updateOutput", updateOutputLua},
  {"SpatialConvolutionCuFFT_accGradParameters", accGradParametersLua},
  {"SpatialConvolutionCuFFT_updateGradInput", updateGradInputLua},
  {"SpatialConvolutionCuFFT_cleanupBuffers", cleanupBuffersLua},
  {"SpatialConvolutionCuFFT_explorePerformance", explorePerformanceLua},
  {nullptr, nullptr},
};
} // namespace

void initSpatialConvolutionCuFFT(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, functions, "nn");
  lua_pop(L,1);
}

}}}  // namespaces

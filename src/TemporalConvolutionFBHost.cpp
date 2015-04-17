// Copyright 2014 Facebook

#include "cuda/DeviceTensor.cuh"
#include "THC.h"
#include "THCTensor.h"
#include "Utils.h"
#include "CuBLASWrapper.h"
#include "ConvolutionBias.cuh"
#include "DeviceTensorUtils.h"
#include "util/Misc.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <folly/Hash.h>
#include <folly/Memory.h>
#include <glog/logging.h>
#include <iomanip>
#include <luaT.h>
#include <lua.hpp>
#include <string>
#include <sstream>
#include <thpp/Tensor.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <unordered_map>
#include <utility>

using namespace facebook::cuda;
using namespace std;
using namespace thpp;

namespace facebook { namespace deeplearning { namespace torch {

#define LOG_TARGET VLOG(3)

namespace {
template <typename T> T ceil(T a, T b) {
  return (a + b - 1) / b;
}

int updateOutput(lua_State *L) {
  THCState* state = getCutorchState(L);
  auto inputTH = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  auto weightTH = (THCudaTensor*)luaT_getfieldcheckudata(
    L, 1, "weight", "torch.CudaTensor");
  auto biasTH = (THCudaTensor*)luaT_getfieldcheckudata(
    L, 1, "bias", "torch.CudaTensor");
  auto outputTH = (THCudaTensor*)luaT_getfieldcheckudata(
    L, 1, "output", "torch.CudaTensor");
  auto kW = luaT_getfieldcheckint(L, 1, "kW");
  auto dW = luaT_getfieldcheckint(L, 1, "dW");

  THAssert(THCudaTensor_checkGPU(state, 4, inputTH, weightTH,
                                 biasTH, outputTH));

  // Remnants from original Torch code
  auto inputTHFrameSize = luaT_getfieldcheckint(L, 1, "inputFrameSize");
  auto dimS = (THCudaTensor_nDimension(state, inputTH) == 3) ?
    1 : 0; // sequence dimension
  auto dimF = (THCudaTensor_nDimension(state, inputTH) == 3) ?
    2 : 1; // feature dimension
  luaL_argcheck(L,
                THCudaTensor_nDimension(state, inputTH) == 2 ||
                THCudaTensor_nDimension(state, inputTH) == 3,
                2,
                "2D or 3D(batch mode) tensor expected");
  luaL_argcheck(L, THCudaTensor_size(state, inputTH, dimF) == inputTHFrameSize,
                2, "invalid inputTH frame size");
  luaL_argcheck(L, THCudaTensor_size(state, inputTH, dimS) >= kW,
                2, "inputTH sequence smaller than kernel size");

  inputTH = THCudaTensor_newContiguous(state, inputTH);
  SCOPE_EXIT {
    THCudaTensor_free(state, inputTH);
  };

  // Init everything to 3-D  and work in 3-D
  DeviceTensor<float, 3> input;
  DeviceTensor<float, 3> weight;
  DeviceTensor<float, 3> output;
  auto inputDim = THCudaTensor_nDimension(state, inputTH);
  auto nInputTHFrame = THCudaTensor_size(state, inputTH, dimS);
  auto outputTHFrameSize = luaT_getfieldcheckint(L, 1, "outputFrameSize");
  if (inputDim == 2) {
    input = torchToDeviceTensor<float, 2>(state, inputTH).upcastOuter<3>();
    { // Remnants from original Torch code
      auto nOutputTHFrame = (nInputTHFrame - kW) / dW + 1;
      THCudaTensor_resize2d(state,
                            outputTH,
                            nOutputTHFrame,
                            outputTHFrameSize);
    }
    output = torchToDeviceTensor<float, 2>(state, outputTH).upcastOuter<3>();
  } else {
    CHECK_EQ(3, inputDim);
    auto nBatchFrame = THCudaTensor_size(state, inputTH, 0);
    input = torchToDeviceTensor<float, 3>(state, inputTH);
    { // Remnants from original Torch code
      auto nOutputTHFrame = (nInputTHFrame - kW) / dW + 1;
      THCudaTensor_resize3d(state,
                            outputTH,
                            nBatchFrame,
                            nOutputTHFrame,
                            outputTHFrameSize);
    }
    output = torchToDeviceTensor<float, 3>(state, outputTH);
  }
  weight = torchToDeviceTensor<float, 3>(state, weightTH);

  auto nBatches = input.getSize(0);
  auto inputHeight = input.getSize(1);
  auto outputHeight = output.getSize(1);
  auto inputWidth = input.getSize(2);
  auto outputWidth = output.getSize(2);

  CHECK(THCudaTensor_isContiguous(state, inputTH));
  CHECK(THCudaTensor_isContiguous(state, outputTH));
  CHECK(THCudaTensor_isContiguous(state, weightTH));

  BLASParameters params;
  params.withTransposeB(CUBLAS_OP_T);

  // 2 cases, if kW != dW we isolate independent bands which can
  // be treated as a single batched sgemm. Otherwise we can perform a single
  // fat sgemm.
  // Consider the following case where kW == 2, dW == 3 and inputHeight == 8
  // Band 0 reads the x from input and skips over the ... rows
  // x x x x x x x x x x x x
  // x x x x x x x x x x x x
  // . . . . . . . . . . . .
  // x x x x x x x x x x x x
  // x x x x x x x x x x x x
  // . . . . . . . . . . . .
  // x x x x x x x x x x x x
  // x x x x x x x x x x x x
  //
  // Band 1 reads the o from input and skips over the ... rows
  // . . . . . . . . . . . .
  // o o o o o o o o o o o o
  // o o o o o o o o o o o o
  // . . . . . . . . . . . .
  // o o o o o o o o o o o o
  // o o o o o o o o o o o o
  // . . . . . . . . . . . .
  // o o o o o o o o o o o o  this is incomplete and thus skipped
  //                          (consistent with the original implementation)
  //
  if (dW != kW) {
    // Compute over numBand non-overlapping independent bands of size kW
    auto numBand = ceil(kW, dW);
    LOG_TARGET << "Using " << numBand << " non-overlapping independent bands";
    for (auto band = 0; band < numBand; ++band) {
      // Each independent band starts off at index 'band' from 0
      thrust::host_vector<float*> outputs;
      thrust::host_vector<const float*> inputs;
      thrust::host_vector<const float*> weights;
      for (auto batch = 0; batch < nBatches; ++batch) {
        outputs.push_back(output[batch][band].data());
        inputs.push_back(input[batch][band * dW].data());
        weights.push_back(weight[0].data());
      }

      // Does at least 1 band fit within both the input and output ?
      auto modif =
        (band + kW <= inputHeight && band < outputHeight) ? 1 : 0;
      // After the first band, how many other fit ?
      auto count = std::min(
        (inputHeight - kW - band) / (dW * numBand) + modif,
        (outputHeight - 1 - band) / (numBand) + modif
      );
      LOG_TARGET << "Band " << band << " has size " << count;

      if (count < 1) {
        continue;
      }
      int sizeWeight[2] {outputWidth, kW * inputWidth};
      int strideWeight[2] {kW * inputWidth, 1};
      DeviceTensor<float, 2> weightModel(nullptr,
                                         sizeWeight,
                                         strideWeight);
      int sizeIn[2] {count, sizeWeight[1]};
      int strideIn[2] {numBand * dW * inputWidth, 1};
      DeviceTensor<float, 2> inputModel(nullptr,
                                        sizeIn,
                                        strideIn);
      int sizeOut[2] {count, outputWidth};
      int strideOut[2] {numBand * outputWidth, 1};
      DeviceTensor<float, 2> outputModel(nullptr,
                                         sizeOut,
                                         strideOut);
      matmultBatched(outputs, inputs, weights,
                     outputModel, inputModel, weightModel,
                     params);
    }
  } else if (dW == kW) {
    // One single nice fat gemm, for instance in cnn-text:
    // 32 x 16 x 512 <- 32 x 32 x 512 * 512 x 512 x 2
    CHECK_EQ(kW, weight.getSize(1));
    CHECK_EQ(inputWidth, inputWidth);
    CHECK_EQ(0, inputHeight % kW);
    int sizeWeight[2] {outputWidth, kW * inputWidth};
    int strideWeight[2] {sizeWeight[1], 1};
    DeviceTensor<float, 2> wei(weight.data(),
                               sizeWeight,
                               strideWeight);
    int sizeIn[2] {nBatches * inputHeight / kW,
        inputWidth * kW};
    int strideIn[2] {sizeIn[1], 1};
    DeviceTensor<float, 2> in(input.data(),
                              sizeIn,
                              strideIn);
    int sizeOut[2] {nBatches * outputHeight, outputWidth};
    int strideOut[2] {sizeOut[1], 1};
    DeviceTensor<float, 2> out(output.data(),
                               sizeOut,
                               strideOut);
    matmult(out, in, wei, params);
  }

  bias::updateOutputTemporalBias(state, outputTH, biasTH);

  return 0;
}

int updateGradInput(lua_State *L) {
  THCState* state = getCutorchState(L);
  auto inputTH = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  auto gradOutputTH = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
  auto kW = luaT_getfieldcheckint(L, 1, "kW");
  auto dW = luaT_getfieldcheckint(L, 1, "dW");
  auto weightTH = (THCudaTensor*)luaT_getfieldcheckudata(
    L, 1, "weight", "torch.CudaTensor");
  auto gradInputTH = (THCudaTensor*)luaT_getfieldcheckudata(
      L, 1, "gradInput", "torch.CudaTensor");

  THAssert(THCudaTensor_checkGPU(state, 4, inputTH, gradOutputTH, weightTH,
                                gradInputTH));

  // Init everything to 3-D  and work in 3-D
  DeviceTensor<float, 3> gradInput;
  DeviceTensor<float, 3> weight;
  DeviceTensor<float, 3> gradOutput;
  auto gradOutputDim = THCudaTensor_nDimension(state, gradOutputTH);
  THCudaTensor_resizeAs(state, gradInputTH, inputTH);
  if (gradOutputDim == 2) {
    auto tmpOut = torchToDeviceTensor<float, 2>(state, gradOutputTH);
    gradOutput = tmpOut.upcastOuter<3>();
    auto tmpIn = torchToDeviceTensor<float, 2>(state, gradInputTH);
    gradInput = tmpIn.upcastOuter<3>();
  } else {
    CHECK_EQ(3, gradOutputDim);
    gradOutput = torchToDeviceTensor<float, 3>(state, gradOutputTH);
    gradInput = torchToDeviceTensor<float, 3>(state, gradInputTH);
  }
  weight = torchToDeviceTensor<float, 3>(state, weightTH);

  auto nBatches = gradInput.getSize(0);
  auto inputHeight = gradInput.getSize(1);
  auto outputHeight = gradOutput.getSize(1);
  auto inputWidth = gradInput.getSize(2);
  auto outputWidth = gradOutput.getSize(2);

  BLASParameters params;
  // 2 cases, if kW != dW we isolate independent bands which can
  // be treated as a single batched sgemm. Otherwise we can perform a single
  // fat sgemm.
  if (dW != kW) {
    // In this case, we accumulate multiple results in gradInput, need to
    // zero it first and accumulate into it.
    params.withAccumulate(true);
    THCudaTensor_zero(state, gradInputTH);

    // Compute over numBand non-overlapping independent bands of size kW
    auto numBand = ceil(kW, dW);
    LOG_TARGET << "Using " << numBand << " non-overlapping independent bands";
    for (auto band = 0; band < numBand; ++band) {
      // Each independent band starts off at index 'band' from 0
      thrust::host_vector<float*> gradInputs;
      thrust::host_vector<const float*> gradOutputs;
      thrust::host_vector<const float*> weights;
      for (auto batch = 0; batch < nBatches; ++batch) {
        gradOutputs.push_back(gradOutput[batch][band].data());
        gradInputs.push_back(gradInput[batch][band * dW].data());
        weights.push_back(weight[0].data());
      }

      // Does at least 1 band fit within both the gradInput and gradOutput ?
      auto modif =
        (band + kW <= inputHeight && band < outputHeight) ?
        1 : 0;
      // After the first band, how many other fit ?
      auto count = std::min(
        (inputHeight - kW - band) / (dW * numBand) + modif,
        (outputHeight - 1 - band) / (numBand) + modif
      );
      LOG_TARGET << "Band " << band << " has size " << count;

      if (count < 1) {
        continue;
      }
      int sizeWeight[2] {outputWidth, kW * inputWidth};
      int strideWeight[2] {kW * inputWidth, 1};
      DeviceTensor<float, 2> weightModel(nullptr,
                                         sizeWeight,
                                         strideWeight);
      int sizeIn[2] {count, sizeWeight[1]};
      int strideIn[2] {numBand * dW * inputWidth, 1};
      DeviceTensor<float, 2> gradInputModel(nullptr,
                                            sizeIn,
                                            strideIn);
      int sizeOut[2] {count, outputWidth};
      int strideOut[2] {numBand * outputWidth, 1};
      DeviceTensor<float, 2> gradOutputModel(nullptr,
                                             sizeOut,
                                             strideOut);
      matmultBatched(gradInputs, gradOutputs, weights,
                     gradInputModel, gradOutputModel, weightModel,
                     params);
    }
  } else if (dW == kW) {
    // One single nice fat gemm, for instance in cnn-text:
    // 32 x 16 x 512 <- 32 x 32 x 512 * 512 x 512 x 2
    CHECK_EQ(kW, weight.getSize(1));
    CHECK_EQ(0, inputHeight % kW);
    int sizeWeight[2] {outputWidth, kW * inputWidth};
    int strideWeight[2] {sizeWeight[1], 1};
    DeviceTensor<float, 2> wei(weight.data(),
                               sizeWeight,
                               strideWeight);
    int sizeIn[2] {nBatches * inputHeight / kW, inputWidth * kW};
    int strideIn[2] {sizeIn[1], 1};
    DeviceTensor<float, 2> in(gradInput.data(),
                              sizeIn,
                              strideIn);
    int sizeOut[2] {nBatches * outputHeight, outputWidth};
    int strideOut[2] {sizeOut[1], 1};
    DeviceTensor<float, 2> out(gradOutput.data(),
                               sizeOut,
                               strideOut);
    matmult(in, out, wei, params);
  }

  return 1;
}

int accGradParameters(lua_State *L) {
  THCState* state = getCutorchState(L);
  auto inputTH = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  auto gradOutputTH = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
  float scale = luaL_optnumber(L, 4, 1);
  auto kW = luaT_getfieldcheckint(L, 1, "kW");
  auto dW = luaT_getfieldcheckint(L, 1, "dW");
  auto gradWeightTH = (THCudaTensor*)luaT_getfieldcheckudata(
      L, 1, "gradWeight", "torch.CudaTensor");
  auto gradBiasTH = (THCudaTensor*)luaT_getfieldcheckudata(
      L, 1, "gradBias", "torch.CudaTensor");

  THAssert(THCudaTensor_checkGPU(state, 4, inputTH, gradOutputTH,
                                 gradWeightTH, gradBiasTH));

  inputTH = THCudaTensor_newContiguous(state, inputTH);
  SCOPE_EXIT {
    THCudaTensor_free(state, inputTH);
  };

  // Init everything to 3-D  and work in 3-D
  DeviceTensor<float, 3> input;
  DeviceTensor<float, 3> gradWeight;
  DeviceTensor<float, 3> gradOutput;
  auto gradOutputDim = THCudaTensor_nDimension(state, gradOutputTH);
  if (gradOutputDim == 2) {
    auto tmpOut = torchToDeviceTensor<float, 2>(state, gradOutputTH);
    gradOutput = tmpOut.upcastOuter<3>();
    auto tmpIn = torchToDeviceTensor<float, 2>(state, inputTH);
    input = tmpIn.upcastOuter<3>();
  } else {
    CHECK_EQ(3, gradOutputDim);
    gradOutput = torchToDeviceTensor<float, 3>(state, gradOutputTH);
    input = torchToDeviceTensor<float, 3>(state, inputTH);
  }
  gradWeight = torchToDeviceTensor<float, 3>(state, gradWeightTH);
  auto nBatches = input.getSize(0);
  auto inputHeight = input.getSize(1);
  auto outputHeight = gradOutput.getSize(1);
  auto inputWidth = input.getSize(2);
  auto outputWidth = gradOutput.getSize(2);

  // Transpose batch x inputHeight x filters ->
  //   filters x batch x inputHeight
  // Then, in a loop, select as gradOutput
  // And transpose back for gemm
  // TODO #5407858: Make a buffer for multi-GPU to avoid blocking
  // TODO #5407858: Completely drop storage when we have working, efficient,
  //   inplace transposition
  auto inputHWBTH =
    THCudaTensor_newWithSize3d(state, nBatches, inputHeight, inputWidth);
  DeviceTensor<float, 3> inputHWB =
    torchToDeviceTensor<float, 3>(state, inputHWBTH);
  SCOPE_EXIT {
    THCudaTensor_free(state, inputHWBTH);
  };
  transpose(input, inputHWB, 1);

  // Output remains batch x outputHeight x filters (i.e. 3 x 2 x 6)

  // Transpose outputHeight x kW x inputHeight ->
  //   kW x inputHeight x utputHeight
  // In a loop, perform gemm on (inputHeight x outputHeight )
  // After loop transpose back to outputHeight x kW x inputHeight
  // TODO #5407858: Make a buffer for multi-GPU to avoid blocking
  // TODO #5407858: Completely drop storage when we have working, efficient,
  //   inplace transposition
  auto gradWeightKIOTH =
    THCudaTensor_newWithSize3d(state, kW, inputWidth, outputWidth);
  DeviceTensor<float, 3> gradWeightKIO =
    torchToDeviceTensor<float, 3>(state, gradWeightKIOTH);
  SCOPE_EXIT {
    THCudaTensor_free(state, gradWeightKIOTH);
  };
  // No need to zero since we do not accumulate in this implementation

  BLASParameters params;
  params.withTransposeA(CUBLAS_OP_T).withScaleReal(scale);

  // AccGradParameters does a scatter read from input and convolves with output
  // We need to extract that data because it does not efficiently fit the mxm
  // or batch mxm API.
  //   batch x outputHeight x filters (i.e. 64 x 16 x 512)
  // No case distinction dW != kW vs dW == kW here, it's all a few fat gemms
  for (auto weightCol = 0; weightCol < kW; ++weightCol) {
    // One weightCol at a time from kW x outputHeight x inputHeight
    int sizeGradWeight[2] {inputWidth, outputWidth};
    int strideGradWeight[2] {outputWidth, 1};
    DeviceTensor<float, 2> gradWeightModel(
      gradWeightKIO[weightCol].data(),
      sizeGradWeight,
      strideGradWeight);

    // One weightCol at a time from:
    //   (dW * outputCol + weightCol) x inputWidth x batch
    // We want 'outputH' scattered rows from input with step dW.
    // This can be obtained, without copying, with the following view.
    auto inputoHWBBand = THCudaTensor_new(state);
    THCudaTensor_setStorage3d(
      state,
      inputoHWBBand,
      THCudaTensor_storage(state, inputHWBTH),
      // Take only scattered weightCol + dW * outputCol so shift by
      // appropriate amount
      weightCol * inputWidth * nBatches,
      // Only take a scattered number of rows equal to outputHeight
      outputHeight,               // size[0]
      inputWidth * nBatches * dW, // stride[0]
      inputWidth,                 // size[1]
      nBatches,                   // stride[1]
      nBatches,                   // size[2]
      1                           // stride[2]
    );
    SCOPE_EXIT {
      THCudaTensor_free(state, inputoHWBBand);
    };

    // Extract subtensor
    // TODO #5407858: Make a buffer for multi-GPU to avoid blocking
    auto inputoHWBTH =
      THCudaTensor_newWithSize3d(state, outputHeight, inputWidth, nBatches);
    DeviceTensor<float, 3> inputoHWB =
      torchToDeviceTensor<float, 3>(state, inputoHWBTH);
    SCOPE_EXIT {
      THCudaTensor_free(state, inputoHWBTH);
    };
    THCudaTensor_copy(state, inputoHWBTH, inputoHWBBand);

    // TODO #5407858: Make a buffer for multi-GPU to avoid blocking
    // TODO #5407858: Completely drop storage when we have working, efficient,
    // inplace transposition
    auto inputBoHWTH =
      THCudaTensor_newWithSize3d(state, nBatches, outputHeight, inputWidth);
    DeviceTensor<float, 3> inputBoHW =
      torchToDeviceTensor<float, 3>(state, inputBoHWTH);
    SCOPE_EXIT {
      THCudaTensor_free(state, inputBoHWTH);
    };

    // Transpose to band x outputH x inputW
    // Permute the sizes / strides first to satisfy the invariants required by
    // the tranpose function.
    auto perm = vector<int>({1, 2, 0});
    inputBoHW.permuteDims(perm);
    transpose(inputoHWB, inputBoHW, 2);

    int sizeIn[2] {nBatches * outputHeight, inputWidth};
    int strideIn[2] {inputWidth, 1};
    DeviceTensor<float, 2> inputModel(
      inputBoHW.data(),
      sizeIn,
      strideIn);

    // All gradOutput at a time from outputCol x batch x outputHeight
    int sizeOut[2] {nBatches * outputHeight, outputWidth};
    int strideOut[2] {outputWidth, 1};
    DeviceTensor<float, 2> gradOutputModel(
      gradOutput.data(),
      sizeOut,
      strideOut);
    CHECK_EQ(nBatches, inputBoHW.getSize(0));

    // Lovely fat gemm
    // I x O <- B x oH x I . B x oH x O
    matmult(gradWeightModel, inputModel, gradOutputModel, params);
  }

  // gradWeightKIO is in kW x inputWidth x outputWidth
  // need to transpose it into gradWeight as outputWidth x kW x inputWidth
  // Permute the sizes / strides first to satisfy the invariants required by
  // the tranpose function.
  auto perm = vector<int>({1, 2, 0});
  gradWeight.permuteDims(perm);
  transpose(gradWeightKIO, gradWeight, 2);

  bias::accGradParametersTemporalBias(state, gradOutputTH, gradBiasTH, scale);

  return 0;
}

const luaL_Reg functions [] = {
  {"TemporalConvolutionFB_updateOutput", updateOutput},
  {"TemporalConvolutionFB_updateGradInput", updateGradInput},
  {"TemporalConvolutionFB_accGradParameters", accGradParameters},
  {nullptr, nullptr}
};

}  // namespace

void initTemporalConvolutionFB(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, functions, "nn");
  lua_pop(L,1);
}

}}}  // namespaces

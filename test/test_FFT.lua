-- Copyright 2004-present Facebook. All Rights Reserved.
require('fb.luaunit')
require('fbtorch')

require 'cunn'
require 'fbcunn'
require 'cutorch'
require 'math'

torch.setnumthreads(6)
torch.setdefaulttensortype('torch.FloatTensor')

local mytester = torch.Tester()

local precision = 1e-4

local test = {}
local silence = true
local printResults = false
local printMemory = false
local timeResults = false
local skipTest = false

local maxSize = 1e30
local maxBatch = 4
local maxInputPlanes = 13
local maxOutputPlanes = 13
local maxKernelSize = 7
local maxInputSize = 32 - maxKernelSize


local function reportAndFree(net)
   if (printResults or printMemory) and not silence then
      local free, total = cutorch.getMemoryUsage()
      print('Pre Collect Memory: ' , free , ' free ' , total , ' total',
            total - free, 'consumption')
   end
   -- release entries from the global buffer table
   if net then
      net:cleanupBuffers()
      net = nil
   end
   collectgarbage()
   collectgarbage()
   if (printResults or printMemory) and not silence then
      local free, total = cutorch.getMemoryUsage()
      print('Post Collect Memory: ' , free , ' free ' , total , ' total',
            total - free, 'consumption')
   end
end

local function timeFunction(
      printString, fun, module, arg1, arg2, arg3, arg4, arg5)
   if not timeResults then
      return fun(module, arg1, arg2, arg3, arg4, arg5)
   end

   local numTrials = 5
   local time = 0
   for i = 1, numTrials do
      local timer = torch.Timer()
      cutorch.synchronize()
      fun(module, arg1, arg2, arg3, arg4, arg5)
      cutorch.synchronize()
      if i > 1 then
         time = time + timer:time().real
      end
   end
   time = time / (numTrials - 1)
   if not silence then
      print(printString .. time * 1000 .. ' ms')
   end

   -- Avoid messing up the accGradParameters case, this is benchmarking
   -- only so we're ok
   module.gradBias:zero()
   module.gradWeight:zero()
   return fun(module, arg1, arg2, arg3, arg4, arg5)
end

local function testLoop(problemSize, fftImplementation)
   local batchSize = problemSize[1] or 4 * torch.random(maxBatch)
   local nInputPlanes = problemSize[2] or torch.random(maxInputPlanes)
   local nOutputPlanes = problemSize[3] or torch.random(maxOutputPlanes)
   local kH = problemSize[4] or torch.random(maxKernelSize)
   -- If not specified, make it square to avoid blatant rectangular
   -- inefficiences with FBFFT atm
   local kW = problemSize[5] or torch.random(maxKernelSize)
   local iH = problemSize[6] or
              math.max(kH, torch.random(maxInputSize) + 4 - kH + 1)
   -- If not specified, make it square to avoid blatant rectangular
   -- inefficiences with FBFFT atm
   local iW = problemSize[7] or
              math.max(kW, torch.random(maxInputSize) + 4 - kW + 1)
   local padH = problemSize[8] or math.min(torch.random(5) % 5, kH - 1)
   local padW = problemSize[9] or math.min(torch.random(5) % 5, kW - 1)
   local tileH = problemSize[10]
   local tileW = problemSize[11]
   local reuseRandom = math.min(torch.random(5) % 5 + 1)
   local reuses = {
      nn.SpatialConvolutionFFT.memoryReuseNone,
      nn.SpatialConvolutionFFT.memoryReuseInput,
      nn.SpatialConvolutionFFT.memoryReuseWeight,
      nn.SpatialConvolutionFFT.memoryReuseOutput,
      nn.SpatialConvolutionFFT.memoryReuseAll,
   }
   local reuse = problemSize[12] or reuses[reuseRandom]

   if fftImplementation == 'cufft' then
      iW = iW + 2 * padW
      iH = iH + 2 * padH
      padW = 0
      padH = 0
      tileW = nil
      tileH = nil
   end

   -- Only small tests, having many small random tests that also
   -- exercise synchronizations is far more valuable than bigger ones
   if iW * iH * batchSize * nInputPlanes > maxSize then
     return
   end
   if iW * iH * nOutputPlanes * nInputPlanes > maxSize then
     return
   end
   if iW * iH * batchSize * nOutputPlanes > maxSize then
     return
   end

   local scale = torch.random(100) / 100.0
   if not silence then
      print('Running ', batchSize, nInputPlanes, nOutputPlanes,
            kH, kW, iH, iW, scale, ' pad by ', padH, 'x', padW,
            ' tile by ', tileH, 'x', tileW, ' reuse ', reuse)
   end

   local input = torch.CudaTensor(batchSize, nInputPlanes, iH, iW):normal()
   local gradOutput = torch.CudaTensor(batchSize,
                                       nOutputPlanes,
                                       iH + 2 * padH - kH + 1,
                                       iW + 2 * padW - kW + 1):normal()

   local netCuDNN, output, gradInput, gradWeight, gradBias
   -- Convenient way to skip tests to debug performance
   if not skipTest then
      netCuDNN =
         cudnn.SpatialConvolution(nInputPlanes, nOutputPlanes,
                                  kW, kH, 1, 1, padW, padH):cuda()
      netCuDNN.gradWeight:zero()
      netCuDNN.gradBias:zero()

      output =
         timeFunction('CUDNN updateOutput: ', netCuDNN.updateOutput,
                      netCuDNN, input, scale):float()
      gradInput =
         timeFunction('CUDNN updateGradInput: ', netCuDNN.updateGradInput,
                      netCuDNN, input, gradOutput):float()
      timeFunction('CUDNN accGradParameters: ', netCuDNN.accGradParameters,
                   netCuDNN, input, gradOutput, scale)
      gradWeight = netCuDNN.gradWeight:float()
      gradBias = netCuDNN.gradBias:float()
   end

   local net
   if tileH and tileW then
      net =
         nn.SpatialConvolutionFFTTiled(nInputPlanes,
                                       nOutputPlanes,
                                       kW,
                                       kH,
                                       1,
                                       1,
                                       padW,
                                       padH,
                                       tileW,
                                       tileH,
                                       reuse)
   else
      if fftImplementation == 'fbfft' then
         net = nn.SpatialConvolutionFBFFT(
            nInputPlanes, nOutputPlanes, kW, kH, 1, 1, padW, padH, reuse)
      elseif fftImplementation == 'cufft' then
         net = nn.SpatialConvolutionCuFFT(
            nInputPlanes, nOutputPlanes, kW, kH, 1, 1, padW, padH, reuse)
      elseif fftImplementation == 'fbfftgemm' then
         net = nn.SpatialConvolutionFBFFTGemm(
            nInputPlanes, nOutputPlanes, kW, kH, 1, 1, padW, padH, reuse)
      else
         assert(false, 'Unknown fftImplementation ' .. fftImplementation)
      end
   end

   local name = fftImplementation
   net:cuda()
   net.gradWeight:zero()
   net.gradBias:zero()
   if netCuDNN then
      net.weight:copy(netCuDNN.weight)
      net.bias:copy(netCuDNN.bias)
   end
   -- net.cudnnDebug = false
   -- net.printDebugLevel = -1

   local outputFFT = timeFunction(name .. 'updateOutput: ',
                                  net.updateOutput,
                                  net,
                                  input):float()

   local gradInputFFT = timeFunction(name .. 'updateGradInput: ',
                                     net.updateGradInput,
                                     net,
                                     input,
                                     gradOutput):float()
   timeFunction(name .. 'accGradParameters: ',
                net.accGradParameters,
                net,
                input,
                gradOutput,
                scale)

   if not skipTest then
      local gradWeightFFT = net.gradWeight:float()
      local gradBiasFFT = net.gradBias:float()

      if printResults and not silence then
         local norm = math.sqrt(output:dot(output) + 1e-8)
         print('updateOutput' .. name, output:dist(outputFFT) / norm)
         local norm = math.sqrt(gradInput:dot(gradInput) + 1e-8)
         print('updateGradInput' .. name,
               gradInput:dist(gradInputFFT) / norm)
         local norm = math.sqrt(gradWeight:dot(gradWeight) + 1e-8)
         print('accGradParameters' .. name .. ' (weight)',
               gradWeight:dist(gradWeightFFT) / norm)
         local norm = math.sqrt(gradBias:dot(gradBias) + 1e-8)
         print('accGradParameters' .. name .. ' (bias)',
               gradBias:dist(gradBiasFFT) / norm)
      end

      local norm = math.sqrt(output:dot(output) + 1e-8)
      mytester:assertle(output:dist(outputFFT) / norm,
                        precision, 'error on output')
      local norm = math.sqrt(gradInput:dot(gradInput) + 1e-8)
      mytester:assertle(gradInput:dist(gradInputFFT) / norm,
                        precision, 'error on gradInput')
      local norm = math.sqrt(gradWeight:dot(gradWeight) + 1e-8)
      mytester:assertle(gradWeight:dist(gradWeightFFT) / norm,
                        precision, 'error on gradWeight')
      local norm = math.sqrt(gradBias:dot(gradBias) + 1e-8)
      mytester:assertle(gradBias:dist(gradBiasFFT) / norm,
                        precision, 'error on gradBias')
   end

   return net
end

-- batch, inputPlanes, outputPlanes, kH, kW, iH, iW
local problemSizes = {
  {1, 1, 1, 1, 1, 1, 1},
  {1, 1, 1, 1, 1, 1, 2},
  {1, 1, 1, 1, 1, 1, 3},
  {1, 1, 1, 3, 3, 4, 4},
  {1, 1, 1, 3, 3, 8, 8},
  {2, 1, 3, 1, 1, 1, 1},
  {2, 3, 1, 1, 1, 1, 1},
  {2, 3, 4, 5, 5, 5, 5},
  {1, 1, 1, 3, 3, 3, 3},
  {1, 1, 1, 2, 2, 2, 2},
  {1, 1, 1, 1, 2, 1, 2},
  {1, 1, 1, 1, 1, 2, 3},
  {2, 3, 4, 5, 5, 5, 5},
  {128, 64, 64, 1, 1, 1, 1},
  {128, 64, 100, 1, 1, 1, 1},
  {128, 64, 64, 3, 3, 3, 3},
  {128, 64, 64, 3, 3, 3, 3},
  {128, 64, 64, 3, 3, 3, 3},
  {128, 64, 64, 3, 3, 3, 3},
  {128, 64, 64, 3, 3, 3, 3},
  {1, 1, 1, 7, 5, 13, 14},
  -- Cannot put in unit tests due to 5GB memory limit
  --  {128, 128, 128, 3, 3, 128, 128}, -- falls back to cudnn
  {1,  1, 1, 5, 5, 27, 27, 0, 0},
  {1,  1, 1, 5, 5, 27, 27, 1, 0},
  {1,  1, 1, 5, 5, 27, 27, 0, 1},
  {1,  1, 1, 5, 5, 27, 27, 1, 2},
  {1,  1, 1, 5, 5, 27, 27, 2, 1},
  {1,  1, 1, 5, 5, 27, 27, 2, 2},
  {1,  1, 1, 3, 4, 19, 23, 0, 0},
  {1,  1, 1, 3, 4, 19, 23, 1, 0},
  {1,  1, 1, 3, 4, 19, 23, 0, 1},
  {1,  1, 1, 3, 4, 19, 23, 1, 2},
  {1,  1, 1, 3, 4, 19, 23, 2, 1},
  {1,  1, 1, 3, 4, 19, 23, 2, 2},
}

local _problemSizesICLR2015 = {
  {16, 16, 16, 3, 3, 13, 13},
  {16, 16, 16, 3, 3, 16, 16},
  {16, 16, 16, 3, 3, 27, 27},
  {16, 16, 16, 3, 3, 32, 32},
  {16, 16, 16, 3, 3, 57, 57},
  {16, 16, 16, 3, 3, 64, 64},
  {32, 32, 32, 3, 3, 13, 13},
  {32, 32, 32, 3, 3, 16, 16},
  {32, 32, 32, 3, 3, 27, 27},
  {32, 32, 32, 3, 3, 32, 32},
  {32, 32, 32, 3, 3, 57, 57},
  {32, 32, 32, 3, 3, 64, 64},
  {64, 64, 64, 3, 3, 13, 13},
  {64, 64, 64, 3, 3, 16, 16},
  {64, 64, 64, 3, 3, 27, 27},
  {64, 64, 64, 3, 3, 32, 32},
  {64, 64, 64, 3, 3, 57, 57},
  {64, 64, 64, 3, 3, 64, 64},
  {128, 128, 128, 3, 3, 13, 13},
  {128, 128, 128, 3, 3, 16, 16},
  {128, 128, 128, 3, 3, 27, 27},
  {128, 128, 128, 3, 3, 32, 32},
  {128, 128, 128, 3, 3, 57, 57},
  {128, 128, 128, 3, 3, 64, 64},
}

local _problemSizesAlexNet = {
   -- 1 GPU
   {128,  96, 256, 5, 5, 27, 27, 2, 2,
    nil, nil, nn.SpatialConvolutionFFT.memoryReuseNone},
   {128,  96, 256, 5, 5, 27, 27, 2, 2,
    nil, nil, nn.SpatialConvolutionFFT.memoryReuseAll},
   {128, 256, 384, 3, 3, 13, 13, 1, 1,
    nil, nil, nn.SpatialConvolutionFFT.memoryReuseNone},
   {128, 256, 384, 3, 3, 13, 13, 1, 1,
    nil, nil, nn.SpatialConvolutionFFT.memoryReuseAll},
   {128, 384, 384, 3, 3, 13, 13, 1, 1,
    nil, nil, nn.SpatialConvolutionFFT.memoryReuseNone},
   {128, 384, 384, 3, 3, 13, 13, 1, 1,
    nil, nil, nn.SpatialConvolutionFFT.memoryReuseAll},
   {128, 384, 256, 3, 3, 13, 13, 1, 1,
    nil, nil, nn.SpatialConvolutionFFT.memoryReuseNone},
   {128, 384, 256, 3, 3, 13, 13, 1, 1,
    nil, nil, nn.SpatialConvolutionFFT.memoryReuseAll},
   -- 2 GPU model parallel
   {128,  48, 128, 5, 5, 27, 27, 2, 2,
    nil, nil, nn.SpatialConvolutionFFT.memoryReuseNone},
   {128,  48, 128, 5, 5, 27, 27, 2, 2,
    nil, nil, nn.SpatialConvolutionFFT.memoryReuseAll},
   {128, 256, 192, 3, 3, 13, 13, 1, 1,
    nil, nil, nn.SpatialConvolutionFFT.memoryReuseNone},
   {128, 256, 192, 3, 3, 13, 13, 1, 1,
    nil, nil, nn.SpatialConvolutionFFT.memoryReuseAll},
   {128, 192, 192, 3, 3, 13, 13, 1, 1,
    nil, nil, nn.SpatialConvolutionFFT.memoryReuseNone},
   {128, 192, 192, 3, 3, 13, 13, 1, 1,
    nil, nil, nn.SpatialConvolutionFFT.memoryReuseAll},
   {128, 192, 128, 3, 3, 13, 13, 1, 1,
    nil, nil, nn.SpatialConvolutionFFT.memoryReuseNone},
   {128, 192, 128, 3, 3, 13, 13, 1, 1,
    nil, nil, nn.SpatialConvolutionFFT.memoryReuseAll},
   -- 4 GPU model parallel
   {128,  24,  64, 5, 5, 27, 27, 2, 2,
    nil, nil, nn.SpatialConvolutionFFT.memoryReuseNone},
   {128,  24,  64, 5, 5, 27, 27, 2, 2,
    nil, nil, nn.SpatialConvolutionFFT.memoryReuseAll},
   {128, 256,  96, 3, 3, 13, 13, 1, 1,
    nil, nil, nn.SpatialConvolutionFFT.memoryReuseNone},
   {128, 256,  96, 3, 3, 13, 13, 1, 1,
    nil, nil, nn.SpatialConvolutionFFT.memoryReuseAll},
   {128,  96,  96, 3, 3, 13, 13, 1, 1,
    nil, nil, nn.SpatialConvolutionFFT.memoryReuseNone},
   {128,  96,  96, 3, 3, 13, 13, 1, 1,
    nil, nil, nn.SpatialConvolutionFFT.memoryReuseAll},
   {128,  96,  64, 3, 3, 13, 13, 1, 1,
    nil, nil, nn.SpatialConvolutionFFT.memoryReuseNone},
   {128,  96,  64, 3, 3, 13, 13, 1, 1,
    nil, nil, nn.SpatialConvolutionFFT.memoryReuseAll},
}

local _problemSizesVGG = {
   {64, 64, 64, 3, 3, 32, 32, 0, 0, 8, 8,
    nn.SpatialConvolutionFFT.memoryReuseNone},
   {64, 64, 64, 3, 3, 32, 32, 0, 0, 8, 8,
    nn.SpatialConvolutionFFT.memoryReuseAll},
   {64, 64, 64, 3, 3, 32, 32, 0, 0, 16, 16,
    nn.SpatialConvolutionFFT.memoryReuseNone},
   {64, 64, 64, 3, 3, 32, 32, 0, 0, 16, 16,
    nn.SpatialConvolutionFFT.memoryReuseAll},
   -- Test fallback to FBFFT convolutions
   {64, 64, 64, 3, 3, 32, 32, 0, 0, 32, 32,
    nn.SpatialConvolutionFFT.memoryReuseNone},
   {64, 64, 64, 3, 3, 32, 32, 0, 0, 32, 32,
    nn.SpatialConvolutionFFT.memoryReuseAll},
   {64, 64, 64, 3, 3, 64, 64, 0, 0, 16, 16,
    nn.SpatialConvolutionFFT.memoryReuseNone},
   {64, 64, 64, 3, 3, 64, 64, 0, 0, 16, 16,
    nn.SpatialConvolutionFFT.memoryReuseAll},
   {64, 64, 64, 3, 3, 64, 64, 0, 0, 32, 32,
    nn.SpatialConvolutionFFT.memoryReuseNone},
   {64, 64, 64, 3, 3, 64, 64, 0, 0, 32, 32,
    nn.SpatialConvolutionFFT.memoryReuseAll},
   {64, 64, 64, 3, 3, 128, 128, 0, 0, 16, 16,
    nn.SpatialConvolutionFFT.memoryReuseNone},
   {64, 64, 64, 3, 3, 128, 128, 0, 0, 16, 16,
    nn.SpatialConvolutionFFT.memoryReuseAll},
   {64, 64, 64, 3, 3, 128, 128, 0, 0, 32, 32,
    nn.SpatialConvolutionFFT.memoryReuseNone},
   {64, 64, 64, 3, 3, 128, 128, 0, 0, 32, 32,
    nn.SpatialConvolutionFFT.memoryReuseAll},
   {64, 64, 128, 3, 3, 112, 112, 1, 1, 16, 16,
    nn.SpatialConvolutionFFT.memoryReuseNone},
   {64, 64, 128, 3, 3, 112, 112, 1, 1, 16, 16,
    nn.SpatialConvolutionFFT.memoryReuseAll},
   {64, 64, 128, 3, 3, 112, 112, 1, 1, 32, 32,
    nn.SpatialConvolutionFFT.memoryReuseNone},
   {64, 64, 128, 3, 3, 112, 112, 1, 1, 32, 32,
    nn.SpatialConvolutionFFT.memoryReuseAll},
}


-- These should correspond with Soumith's benchmarks
-- https://raw.githubusercontent.com/soumith/convnet-benchmarks/master/torch7/imagenet_winners/output_raw.log
local _benchmarkAlexNet = {
   -- 1 GPU
   {128, 64, 192, 5, 5, 27, 27, 2, 2,
    nil, nil, nn.SpatialConvolutionFFT.memoryReuseAll},
   {128, 192, 384, 3, 3, 13, 13, 1, 1,
    nil, nil, nn.SpatialConvolutionFFT.memoryReuseAll},
   {128, 384, 256, 3, 3, 13, 13, 1, 1,
    nil, nil, nn.SpatialConvolutionFFT.memoryReuseAll},
   {128, 256, 256, 3, 3, 13, 13, 1, 1,
    nil, nil, nn.SpatialConvolutionFFT.memoryReuseAll},

   -- 1 GPU
   {128, 64, 192, 5, 5, 27, 27, 2, 2,
    nil, nil, nn.SpatialConvolutionFFT.memoryReuseNone},
   {128, 192, 384, 3, 3, 13, 13, 1, 1,
    nil, nil, nn.SpatialConvolutionFFT.memoryReuseNone},
   {128, 384, 256, 3, 3, 13, 13, 1, 1,
    nil, nil, nn.SpatialConvolutionFFT.memoryReuseNone},
   {128, 256, 256, 3, 3, 13, 13, 1, 1,
    nil, nil, nn.SpatialConvolutionFFT.memoryReuseNone},
}

local _benchmarkOverFeat = {
   -- 1 GPU
   {128, 96, 256, 5, 5, 24, 24, 2, 2,
    nil, nil, nn.SpatialConvolutionFFT.memoryReuseAll},
   {128, 256, 512, 3, 3, 12, 12, 1, 1,
    nil, nil, nn.SpatialConvolutionFFT.memoryReuseAll},
   {128, 512, 1024, 3, 3, 12, 12, 1, 1,
    nil, nil, nn.SpatialConvolutionFFT.memoryReuseAll},
   {128, 1024, 1024, 3, 3, 12, 12, 1, 1,
    nil, nil, nn.SpatialConvolutionFFT.memoryReuseAll},

   -- 1 GPU
   {128, 96, 256, 5, 5, 24, 24, 2, 2,
    nil, nil, nn.SpatialConvolutionFFT.memoryReuseNone},
   {128, 256, 512, 3, 3, 12, 12, 1, 1,
    nil, nil, nn.SpatialConvolutionFFT.memoryReuseNone},
   {128, 512, 1024, 3, 3, 12, 12, 1, 1,
    nil, nil, nn.SpatialConvolutionFFT.memoryReuseNone},
   {128, 1024, 1024, 3, 3, 12, 12, 1, 1,
    nil, nil, nn.SpatialConvolutionFFT.memoryReuseNone},
}

local _benchmarkVGG = {
   -- 1 GPU
   {64, 3, 64, 3, 3, 224, 224, 1, 1,
    32, 32, nn.SpatialConvolutionFFT.memoryReuseAll},
   {64, 64, 128, 3, 3, 112, 112, 1, 1,
    32, 32, nn.SpatialConvolutionFFT.memoryReuseAll},
   {64, 128, 256, 3, 3, 56, 56, 1, 1,
    32, 32, nn.SpatialConvolutionFFT.memoryReuseAll},

   {64, 256, 256, 3, 3, 56, 56, 1, 1,
    32, 32, nn.SpatialConvolutionFFT.memoryReuseAll},

   {64, 256, 512, 3, 3, 28, 28, 1, 1,
    nil, nil, nn.SpatialConvolutionFFT.memoryReuseAll},
   {64, 512, 512, 3, 3, 28, 28, 1, 1,
    nil, nil, nn.SpatialConvolutionFFT.memoryReuseAll},

   {64, 512, 512, 3, 3, 14, 14, 1, 1,
    nil, nil, nn.SpatialConvolutionFFT.memoryReuseAll},
   {64, 512, 512, 3, 3, 14, 14, 1, 1,
    nil, nil, nn.SpatialConvolutionFFT.memoryReuseAll},

   -- 1 GPU
   {64, 3, 64, 3, 3, 224, 224, 1, 1,
    32, 32, nn.SpatialConvolutionFFT.memoryReuseNone},
   {64, 64, 128, 3, 3, 112, 112, 1, 1,
    32, 32, nn.SpatialConvolutionFFT.memoryReuseNone},
   {64, 128, 256, 3, 3, 56, 56, 1, 1,
    32, 32, nn.SpatialConvolutionFFT.memoryReuseNone},

   {64, 256, 256, 3, 3, 56, 56, 1, 1,
    32, 32, nn.SpatialConvolutionFFT.memoryReuseNone},

   {64, 256, 512, 3, 3, 28, 28, 1, 1,
    nil, nil, nn.SpatialConvolutionFFT.memoryReuseNone},
   {64, 512, 512, 3, 3, 28, 28, 1, 1,
    nil, nil, nn.SpatialConvolutionFFT.memoryReuseNone},

   {64, 512, 512, 3, 3, 14, 14, 1, 1,
    nil, nil, nn.SpatialConvolutionFFT.memoryReuseNone},
   {64, 512, 512, 3, 3, 14, 14, 1, 1,
    nil, nil, nn.SpatialConvolutionFFT.memoryReuseNone},
}

local _stressTest = {
   {1, 128, 128, 3, 3, 8, 8, 0, 0,
    nil, nil, nn.SpatialConvolutionFFT.memoryReuseNone},
   {64, 3, 128, 3, 3, 8, 8, 0, 0,
    nil, nil, nn.SpatialConvolutionFFT.memoryReuseNone},
   {64, 3, 512, 3, 3, 8, 8, 0, 0,
    nil, nil, nn.SpatialConvolutionFFT.memoryReuseNone},
   {1, 256, 512, 3, 3, 8, 8, 0, 0,
    nil, nil, nn.SpatialConvolutionFFT.memoryReuseNone},
   {64, 128, 128, 3, 3, 8, 8, 0, 0,
    nil, nil, nn.SpatialConvolutionFFT.memoryReuseNone},
   {64, 256, 512, 3, 3, 8, 8, 0, 0,
    nil, nil, nn.SpatialConvolutionFFT.memoryReuseNone},
   {1, 16, 16, 3, 3, 16, 16, 0, 0,
    nil, nil, nn.SpatialConvolutionFFT.memoryReuseNone},
   {1, 128, 128, 3, 3, 16, 16, 0, 0,
    nil, nil, nn.SpatialConvolutionFFT.memoryReuseNone},
   {1, 256, 512, 3, 3, 16, 16, 0, 0,
    nil, nil, nn.SpatialConvolutionFFT.memoryReuseNone},
   {64, 3, 128, 3, 3, 16, 16, 0, 0,
    nil, nil, nn.SpatialConvolutionFFT.memoryReuseNone},
   {64, 3, 512, 3, 3, 16, 16, 0, 0,
    nil, nil, nn.SpatialConvolutionFFT.memoryReuseNone},
   {64, 128, 128, 3, 3, 16, 16, 0, 0,
    nil, nil, nn.SpatialConvolutionFFT.memoryReuseNone},
   {64, 256, 512, 3, 3, 16, 16, 0, 0,
    nil, nil, nn.SpatialConvolutionFFT.memoryReuseNone},
   {1, 16, 16, 3, 3, 32, 32, 0, 0,
    nil, nil, nn.SpatialConvolutionFFT.memoryReuseNone},
-- Investigation says the cost of FFT weights is too high since
-- they are only used once in this case. Good thing is that batch
-- size of 1 should be for inference only and precomputing the FFT
-- of the weights is a viable approach
   {1, 128, 128, 3, 3, 32, 32, 0, 0,
    nil, nil, nn.SpatialConvolutionFFT.memoryReuseNone},
   {1, 256, 512, 3, 3, 32, 32, 0, 0,
    nil, nil, nn.SpatialConvolutionFFT.memoryReuseNone},
----------------------------------------------------------------
   {64, 3, 128, 3, 3, 32, 32, 0, 0,
    nil, nil, nn.SpatialConvolutionFFT.memoryReuseNone},
   {64, 3, 512, 3, 3, 32, 32, 0, 0,
    nil, nil, nn.SpatialConvolutionFFT.memoryReuseNone},
   {64, 128, 128, 3, 3, 32, 32, 0, 0,
    nil, nil, nn.SpatialConvolutionFFT.memoryReuseNone},
   {64, 256, 512, 3, 3, 32, 32, 0, 0,
    nil, nil, nn.SpatialConvolutionFFT.memoryReuseNone},
}

local problemsToRun = _stressTest
local num_random_configurations = 25

--[[
-- Convenient override of the default that are used for unit tests
problemsToRun = _problemSizesAlexNet
problemsToRun = _problemSizesICLR2015
problemsToRun = _problemSizesVGG
printMemory = true
timeResults = true
num_random_configurations = 0
--]]

local testCuFFT = false
local testFBFFT = true
local testFBFFTGemm = true

function test.test()
   for i = 1, #problemsToRun do
      if testFBFFT then
         local net = testLoop(problemsToRun[i], 'fbfft')
         reportAndFree(net)
      end
      if testFBFFTGemm then
         local net = testLoop(problemsToRun[i], 'fbfftgemm')
         reportAndFree(net)
      end
      if testCuFFT then
         local net = testLoop(problemsToRun[i], 'cufft')
         reportAndFree(net)
      end
   end

   for size in pairs({'big', 'small'}) do
      if size == 'big' then
         maxInputSize = 32 - maxKernelSize
      else
         maxInputSize = 128 - maxKernelSize
      end
      -- random configuration
      for i = 1, num_random_configurations do
         if testFBFFT then
            local net = testLoop({}, 'fbfft')
            reportAndFree(net)
         end
         if testFBFFTGemm then
            local net = testLoop({}, 'fbfftgemm')
            reportAndFree(net)
         end
         if testCuFFT then
            local net = testLoop({}, 'cufft')
            reportAndFree(net)
         end
      end
   end
end

mytester:add(test)
mytester:run()

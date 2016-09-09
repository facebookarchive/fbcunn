-- Copyright 2004-present Facebook. All Rights Reserved.
require('fb.luaunit')
local torch = require('fbtorch')

require 'cunn'
require 'fbcunn'
require 'cutorch'
require 'math'

torch.setnumthreads(6)
torch.setdefaulttensortype('torch.FloatTensor')

local mytester = torch.Tester()

local precision = 1e-4

local test = {}
local printResults = false
local printMemory = false
local timeResults = false

local kNumGPUs = 1
local maxSize = 128000000
local maxBatch = 4
local maxInputPlanes = 13
local maxOutputPlanes = 13
local maxKernelSize = 7
local maxInputSize = 60

local function timeFunction(printString, fun, module, arg1, arg2, arg3)
   if not timeResults then
      return fun(module, arg1, arg2, arg3)
   end

   local numTrials = 5
   local time = 0
   for i = 1, numTrials do
      local timer = torch.Timer()
      cutorch.synchronize()
      fun(module, arg1, arg2, arg3)
      cutorch.synchronize()
      if i > 1 then
         time = time + timer:time().real
      end
   end
   time = time / (numTrials - 1)
   print(printString .. time * 1000 .. " ms")

   -- Avoid messing up the accGradParameters case, this is benchmarking
   -- only so we're ok
   module.gradBias:zero()
   module.gradWeight:zero()
   return fun(module, arg1, arg2, arg3)
end

local function testLoop(problemSize)
   local batchSize = problemSize[1] or 4 * torch.random(maxBatch)
   local nInputPlanes = problemSize[2] or torch.random(maxInputSize)
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

   local padH = 0
   local padW = 0

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
   print('Running ', batchSize, nInputPlanes, nOutputPlanes,
         kH, kW, iH, iW, scale, " pad by ", padH, "x", padW)

   local net =
      cudnn.SpatialConvolution(nInputPlanes, nOutputPlanes,
                               kW, kH, 1, 1, padW, padH):cuda()
   local input = torch.CudaTensor(batchSize, nInputPlanes, iH, iW):normal()
   local gradOutput = torch.CudaTensor(batchSize,
                                       nOutputPlanes,
                                       iH + 2 * padH - kH + 1,
                                       iW + 2 * padW - kW + 1):normal()
   net.gradWeight:zero()
   net.gradBias:zero()

   local output = timeFunction("CUDNN updateOutput: ",
                                net.updateOutput, net, input, scale):float()
   local gradInput =
      timeFunction("CUDNN updateGradInput: ",
                   net.updateGradInput, net, input, gradOutput):float()
   timeFunction("CUDNN accGradParameters: ",
                net.accGradParameters, net, input, gradOutput, scale)
   local gradWeight = net.gradWeight:float()
   local gradBias = net.gradBias:float()

   local netCuFFT = {}
   local outputCuFFT = {}
   local gradInputCuFFT = {}
   local gradWeightCuFFT = {}
   local gradBiasCuFFT = {}

   for k = 1, kNumGPUs do -- Across kNumGPUs GPUs
     if k > 1 then
        cutorch.setDevice(k)
     end

     netCuFFT[k] =
       nn.SpatialConvolutionCuFFT(nInputPlanes, nOutputPlanes,
                                  kW, kH, 1, 1, padW, padH)
     netCuFFT[k].cudnnDebug = true
     netCuFFT[k].gradWeight:zero()
     netCuFFT[k].gradBias:zero()
     netCuFFT[k].weight:copy(net.weight)
     netCuFFT[k].bias:copy(net.bias)
     netCuFFT[k]:cuda()

     outputCuFFT[k] = timeFunction("CuFFT updateOutput: ",
                                   netCuFFT[k].updateOutput,
                                   netCuFFT[k],
                                   input,
                                   scale):float()
     gradInputCuFFT[k] = timeFunction("CuFFT updateGradInput: ",
                                      netCuFFT[k].updateGradInput,
                                      netCuFFT[k],
                                      input,
                                      gradOutput):float()
     timeFunction("CuFFT accGradParameters: ",
                  netCuFFT[k].accGradParameters,
                  netCuFFT[k],
                  input,
                  gradOutput,
                  scale)

--[[
     gradInputCuFFT[k] = timeFunction("CuFFT backward: ",
                                      netCuFFT[k].backward,
                                      netCuFFT[k],
                                      input,
                                      gradOutput,
                                      scale):float()
--]]

     gradWeightCuFFT[k] = netCuFFT[k].gradWeight:float()
     gradBiasCuFFT[k] = netCuFFT[k].gradBias:float()

     if printResults then
        print("Padding WxH = ", padW, "x", padH)
        local norm = math.sqrt(output:dot(output) + 1e-8)
        print("updateOutputCuFFT", output:dist(outputCuFFT[k]) / norm)
        local norm = math.sqrt(gradInput:dot(gradInput) + 1e-8)
        print("updateGradInputCuFFT",
              gradInput:dist(gradInputCuFFT[k]) / norm)
        local norm = math.sqrt(gradWeight:dot(gradWeight) + 1e-8)
        print("accGradParametersCuFFT (weight)",
              gradWeight:dist(gradWeightCuFFT[k]) / norm)
        local norm = math.sqrt(gradBias:dot(gradBias) + 1e-8)
        print("accGradParametersCuFFT (bias)",
              gradBias:dist(gradBiasCuFFT[k]) / norm)
     end

     local norm = math.sqrt(output:dot(output) + 1e-8)
     mytester:assertle(output:dist(outputCuFFT[k]) / norm,
       precision, 'error on output')
     local norm = math.sqrt(gradInput:dot(gradInput) + 1e-8)
     mytester:assertle(gradInput:dist(gradInputCuFFT[k]) / norm,
       precision, 'error on gradInput')
     local norm = math.sqrt(gradWeight:dot(gradWeight) + 1e-8)
     mytester:assertle(gradWeight:dist(gradWeightCuFFT[k]) / norm,
       precision, 'error on gradWeight')
     local norm = math.sqrt(gradBias:dot(gradBias) + 1e-8)
     mytester:assertle(gradBias:dist(gradBiasCuFFT[k]) / norm,
       precision, 'error on gradBias')
  end

   return netCuFFT
end

-- batch, inputPlanes, outputPlanes, kH, kW, iH, iW
local problemSizes = {
  {1, 1, 1, 1, 1, 1, 1},
  {1, 1, 1, 1, 1, 1, 2},
  {1, 1, 1, 1, 1, 1, 3},
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
  {128,  96, 256, 5, 5, 31, 31},
  {128, 256, 384, 3, 3, 15, 15},
  {128, 384, 384, 3, 3, 15, 15},
  {128, 384, 256, 3, 3, 15, 15},
  -- 2 GPU model parallel
  {128,  48, 128, 5, 5, 31, 31},
  {128, 256, 192, 3, 3, 15, 15},
  {128, 192, 192, 3, 3, 15, 15},
  {128, 192, 128, 3, 3, 15, 15},
  -- 4 GPU model parallel
  {128,  24,  64, 5, 5, 31, 31},
  {128, 256,  96, 3, 3, 15, 15},
  {128,  96,  96, 3, 3, 15, 15},
  {128,  96,  64, 3, 3, 15, 15},
}

local function reportAndFree(net)
   if printResults or printMemory then
      local free, total = cutorch.getMemoryUsage()
      print("Pre Collect Memory: " , free , " free " , total , " total")
   end
   assert(torch.type(net) == 'table', torch.type(net))
   -- Kill the local references to, as well as the global buffers
   for i, v in ipairs(net) do
      v:cleanupBuffers()
   end
   collectgarbage()
   collectgarbage()
   if printResults or printMemory then
      local free, total = cutorch.getMemoryUsage()
      print("Post Collect Memory: " , free , " free " , total , " total")
   end
end

local num_random_configurations = 100
local problemsToRun = problemSizes

--[[
-- Convenient override of the default that are used for unit tests
local problemsToRun = _problemSizesAlexNet
local problemsToRun = _problemSizesICLR2015
printMemory = true
timeResults = true
printResults = true

num_random_configurations = 0
printMemory = true
timeResults = true
--]]

function test.test()
  for i = 1, #problemsToRun do
      local net = testLoop(problemsToRun[i])
      reportAndFree(net)
  end
  -- random configuration
  for i = 1, num_random_configurations do
      local net = testLoop({})
      reportAndFree(net)
  end
end

mytester:add(test)
mytester:run()

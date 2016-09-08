require 'cunn'
require 'fbcunn'
require 'math'

require 'fb.luaunit'
require('fbtorch')
g_mytester = torch.Tester()
local fb_test = {}

local silence = true
local printMemory = false
local inferenceOnly = false

local function reportAndFree(net)
   if printMemory and not silence then
      local free, total = cutorch.getMemoryUsage()
      print("Pre Collect Memory: " , free , " free " , total , " total")
   end
   -- release entries from the global buffer table
   if net and net.cleanupBuffers then
      net:cleanupBuffers()
      net = nil
   end
   collectgarbage()
   collectgarbage()
   if printMemory and not silence  then
      local free, total = cutorch.getMemoryUsage()
      print("Post Collect Memory: " , free , " free " , total , " total")
   end
end

local function testSpatialConvolutionTuned(problem, FFTConvolutionClass)
   local batches = problem[1]
   local inputPlanes = problem[2]
   local outputPlanes = problem[3]
   local iH = problem[4]
   local iW = problem[5]
   local kH = problem[6]
   local kW = problem[7]
   local padH = problem[8]
   local padW = problem[9]

   if not silence then
      print('Running ', batches, inputPlanes, outputPlanes,
            " kH = ", kH, " x ", "kW = ", kW,
            " x ", "iH = ", iH, " x ", "iW = ", iW,
            " x ", "padH = ", padH, " x ", padW)
   end

   -- All the necessary checks are already performed while searching
   -- for the best convolution
   local netForward = fbnn.SpatialConvolution(
      inputPlanes,
      outputPlanes,
      kW,
      kH,
      1,
      1,
      padW,
      padH,
      nil,   -- no memory limit
      inferenceOnly  -- not just inference
   )
   if not silence then
      netForward.reportLevel = 2
   end

   local ps = {batches, inputPlanes, iH, iW}
   local input = torch.Tensor(torch.LongStorage(ps)):cuda()
   local ps = {batches,
               outputPlanes,
               iH - kH + 2 * padH + 1,
               iW - kW + 2 * padW + 1}
   local gradOutput = torch.Tensor(torch.LongStorage(ps)):cuda()
   local scale = torch.random(100) / 100.0
   netForward:updateOutput(input)
   if not inferenceOnly then
      netForward:updateGradInput(input, gradOutput)
      netForward:accGradParameters(input, gradOutput, scale)
   end

   return netForward
end


local problemsToRun = {
   -- batch, input, output, iH, iW, kH, kW, padH, padW
   {1, 1, 1, 4, 4, 3, 3, 0, 0},
   {1, 1, 1, 1, 1, 1, 1, 0, 0},
   {1, 1, 1, 1, 2, 1, 2, 0, 0},
   {1, 1, 1, 1, 3, 1, 3, 0, 0},
   {1, 1, 1, 6, 6, 4, 4, 0, 0},
   {1, 1, 1, 11, 11, 8, 8, 0, 0},
   {2, 1, 3, 1, 1, 1, 1, 0, 0},
   {2, 3, 1, 1, 1, 1, 1, 0, 0},
   {2, 3, 4, 5, 5, 5, 5, 0, 0},
   {1, 1, 1, 3, 3, 3, 3, 0, 0},
   {1, 1, 1, 2, 2, 2, 2, 0, 0},
   {1, 1, 1, 1, 2, 1, 2, 0, 0},
   {1, 1, 1, 2, 3, 2, 3, 0, 0},
   {2, 3, 4, 5, 5, 5, 5, 0, 0},
   {128, 64, 64, 1, 1, 1, 1, 0, 0},
   {128, 64, 100, 1, 1, 1, 1, 0, 0},
   {128, 64, 64, 3, 3, 3, 3, 0, 0},
   {128, 64, 64, 3, 3, 3, 3, 0, 0},
   {128, 64, 64, 3, 3, 3, 3, 0, 0},
   {128, 64, 64, 3, 3, 3, 3, 0, 0},
   {128, 64, 64, 3, 3, 3, 3, 0, 0},
   {1, 1, 1, 20, 17, 13, 14, 0, 0},
   -- Cannot put in unit tests due to 5GB memory limit
   --  {128, 128, 128, 128, 128, 3, 3, 0, 0}, -- falls back to cudnn
   {1,  1, 1, 27, 27, 5, 5, 0, 0},
   {1,  1, 1, 27, 27, 5, 5, 1, 0},
   {1,  1, 1, 27, 27, 5, 5, 0, 1},
   {1,  1, 1, 27, 27, 5, 5, 1, 2},
   {1,  1, 1, 27, 27, 5, 5, 2, 1},
   {1,  1, 1, 27, 27, 5, 5, 2, 2},
   {1,  1, 1, 19, 23, 3, 4, 0, 0},
   {1,  1, 1, 19, 23, 3, 4, 1, 0},
   {1,  1, 1, 19, 23, 3, 4, 0, 1},
   {1,  1, 1, 19, 23, 3, 4, 1, 2},
   {1,  1, 1, 19, 23, 3, 4, 2, 1},
   {1,  1, 1, 19, 23, 3, 4, 2, 2},

   {1, 1, 1, 128, 128, 3, 3, 0, 0},
}

local _expensiveProblemsToRun = {
   {1, 512, 768, 16, 16, 14, 14, 0, 0},
   {2, 512, 768, 16, 16, 14, 14, 0, 0},
   {8, 512, 768, 16, 16, 14, 14, 0, 0},
   {1, 512, 768, 24, 24, 14, 14, 0, 0},
   {2, 512, 768, 24, 24, 14, 14, 0, 0},
   {8, 512, 768, 24, 24, 14, 14, 0, 0},
   {1, 512, 768, 72, 72, 14, 14, 0, 0},
   {2, 512, 768, 72, 72, 14, 14, 0, 0},
   {8, 512, 768, 72, 72, 14, 14, 0, 0},
}

local _benchmark3x3 = {
   {64,   3,  64, 224, 224, 3, 3, 1, 1},
   {32, 32, 32, 30, 30, 3, 3, 0, 0},
   {64, 64, 64, 30, 30, 3, 3, 0, 0},
   {128, 128, 128, 30, 30, 3, 3, 0, 0},
   {32, 32, 32, 27, 27, 3, 3, 1, 1},
   {64, 64, 64, 27, 27, 3, 3, 1, 1},
   {128, 128, 128, 27, 27, 3, 3, 1, 1},
   {32, 32, 32, 14, 14, 3, 3, 0, 0},
   {64, 64, 64, 14, 14, 3, 3, 0, 0},
   {128, 128, 128, 14, 14, 3, 3, 0, 0},
   {32, 32, 32, 12, 12, 3, 3, 1, 1},
   {64, 64, 64, 12, 12, 3, 3, 1, 1},
   {128, 128, 128, 12, 12, 3, 3, 1, 1},
   {64, 128, 128,  14,  14, 3, 3, 1, 1},
   {64, 256, 256,  14,  14, 3, 3, 1, 1},
   {64, 512, 512,  14,  14, 3, 3, 1, 1},
}

-- These should correspond with Soumith's benchmarks
-- https://raw.githubusercontent.com/soumith/convnet-benchmarks/master/torch7/imagenet_winners/output_raw.log
local _benchmarkAlexNet = {
   -- 1 GPU
   {128,  64, 192, 27, 27, 5, 5, 2, 2},
   {128, 192, 384, 13, 13, 3, 3, 1, 1},
   {128, 384, 256, 13, 13, 3, 3, 1, 1},
   {128, 256, 256, 13, 13, 3, 3, 1, 1},
}

local _benchmarkOverFeat = {
   -- 1 GPU
   {128,   96,  256, 24, 24, 5, 5, 2, 2},
   {128,  256,  512, 12, 12, 3, 3, 1, 1},
   {128,  512, 1024, 12, 12, 3, 3, 1, 1},
   {128, 1024, 1024, 12, 12, 3, 3, 1, 1},
}

local _benchmarkVGG = {
   -- 1 GPU
   {64,   3,  64, 224, 224, 3, 3, 1, 1},
   {64,  64, 128, 112, 112, 3, 3, 1, 1},
   {64, 128, 256,  56,  56, 3, 3, 1, 1},
   {64, 256, 256,  56,  56, 3, 3, 1, 1},
   {64, 256, 512,  28,  28, 3, 3, 1, 1},
   {64, 512, 512,  28,  28, 3, 3, 1, 1},
   {64, 512, 512,  14,  14, 3, 3, 1, 1},
   {64, 512, 512,  14,  14, 3, 3, 1, 1},
}

--[[
   Uncomment this for expensive problems
   problemsToRun = _expensiveProblemsToRun
   problemsToRun = _benchmarkAlexNet
   problemsToRun = _benchmarkOverFeat
   problemsToRun = _benchmarkVGG
   problemsToRun = _benchmark3x3
   inferenceOnly = true
--]]

function fb_test.testSpatialConvolutionTuned()
   for i = 1, #problemsToRun do
      local net =
         testSpatialConvolutionTuned(problemsToRun[i])
      reportAndFree(net)
   end
end

g_mytester = torch.Tester()
g_mytester:add(fb_test)
g_mytester:run()

require 'cunn'
require 'fbcunn'
require 'math'

require 'fb.luaunit'
require('fbtorch')
g_mytester = torch.Tester()
local fb_test = {}

local silence = true
local timeResults = false
local printDebug = false
local printMemory = false
local testCuDNN = true
local runUpdateOutput = true
local runUpdateGradInput = true
local runAccGradParameters = true

local function reportAndFree(net)
   if printMemory then
      local free, total = cutorch.getMemoryUsage()
      if not silence then
         print('Pre Collect Memory: ' , free , ' free ' , total , ' total')
      end
   end
   -- release entries from the global buffer table
   if net then
      net:cleanupBuffers()
      net = nil
   end
   collectgarbage()
   collectgarbage()
   if printMemory then
      local free, total = cutorch.getMemoryUsage()
      if not silence then
         print('Post Collect Memory: ' , free , ' free ' , total , ' total')
      end
   end
end

local function testTiledFFT(problem, FFTConvolutionClass)
   local batches = problem[1] or torch.random(16)
   local inputPlanes = problem[2] or torch.random(16)
   local outputPlanes = problem[3] or torch.random(16)
   -- Values that make sense, start from kernel size
   local kH = problem[6] or 4 + math.random(11)
   local kW = problem[7] or 4 + math.random(11)
   local iH = problem[4] or 1 + 2 * kH + math.random(13)
   local iW = problem[5] or 1 + 2 * kW + math.random(13)
   local tileH = kH + math.random(5)
   tileH = problem[8] or math.min(tileH, iH - 1)
   local tileW = kW + math.random(5)
   tileW = problem[9] or math.min(tileW, iW - 1)
   local padH = problem[10] or math.min(kH - 1, tileH - kH, math.random(7))
   local padW = problem[11] or math.min(kW - 1, tileW - kW, math.random(7))
   local reuseRandom = math.min(torch.random(5) % 5 + 1)
   local reuses = {
      nn.SpatialConvolutionFFT.memoryReuseNone,
      nn.SpatialConvolutionFFT.memoryReuseInput,
      nn.SpatialConvolutionFFT.memoryReuseWeight,
      nn.SpatialConvolutionFFT.memoryReuseOutput,
      nn.SpatialConvolutionFFT.memoryReuseAll,
   }
   local reuse = problem[12] or reuses[reuseRandom]

   if not silence then
      print('Running ', batches, inputPlanes, outputPlanes,
            ' kH = ', kH, ' x ', 'kW = ', kW,
            ' x ', 'iH = ', iH, ' x ', 'iW = ', iW,
            ' x ', 'padH = ', padH, ' x ', padW, ' tile by ', tileH, 'x', tileW,
            ' reuse = ', reuse)
   end

   -- Testing tiling, 1 batch, input plane, output plane are enough
   local ps = {batches, inputPlanes, iH, iW}
   local input = torch.Tensor(torch.LongStorage(ps)):cuda():normal()
   local ps = {batches,
               outputPlanes,
               iH - kH + 2 * padH + 1,
               iW - kW + 2 * padW + 1}
   local gradOutput = torch.Tensor(torch.LongStorage(ps)):cuda():normal()
   local scale = torch.uniform()
   local net = FFTConvolutionClass(inputPlanes,
                                   outputPlanes,
                                   kW,
                                   kH,
                                   1,
                                   1,
                                   padW,
                                   padH,
                                   tileW,
                                   tileH,
                                   reuse):cuda()
   net.cudnnDebug = testCuDNN -- this line activates internal testing vs CuDNN

   if silence then
      net.reportErrors = false
   end

   if runUpdateOutput then
      net.printDebugLevel = -1
      if net.printDebugLevel >= 3 then
         -- Nasty debugging to be expected
         local val = 1
         input:apply(function() val = val + 1 return val end)
         local val = 1
         net.weight:apply(function() val = val + 1 return val end)
      end

      net:updateOutput(input)
   end


   if runUpdateGradInput then
      net.printDebugLevel = -1
      if net.printDebugLevel >= 3 then
         -- Nasty debugging to be expected
         local val = 1
         gradOutput:apply(function() val = val + 1 return val end)
         local val = 1
         net.weight:apply(function() val = val + 1 return val end)
      end

      net:updateGradInput(input, gradOutput)
   end


   if runAccGradParameters then
      net.printDebugLevel = -1
      if net.printDebugLevel >= 3 then
         -- Nasty debugging to be expected
         scale = 1.0
         local val = 1
         input:apply(function() val = val + 1 return val end)
         local val = 1
         gradOutput:apply(function() val = val + 1 return val end)
      end
      net:accGradParameters(input, gradOutput, scale)
   end

   g_mytester:assert(net.cudnnChecks)

   return net
end


local problemsToRun = {
   -- iH, iW, kH, kW, tileH, tileW, padH, padW, reuse
   {2, 2, 2, 12, 12, 3, 3, 8, 8, 0, 0,
    nn.SpatialConvolutionFFT.memoryReuseNone},
   {2, 2, 2, 128, 128, 3, 3, 16, 16, 0, 0,
    nn.SpatialConvolutionFFT.memoryReuseNone},
   {64, 64, 128, 112, 112, 3, 3, 32, 32, 0, 0,
    nn.SpatialConvolutionFFT.memoryReuseAll},
}

local numTests = 25

-- Convenient override of the default that are used for unit tests
-- numTests = 1
-- silence = false
-- timeResults = true
-- printDebug = false
-- printMemory = false
-- runUpdateOutput = true
-- runUpdateGradInput = true
-- runAccGradParameters = true

local testSync = true
local testAsync = true
local testIterated = true
function fb_test.testTiledFFT()
   for i = 1, #problemsToRun do
      if testSync then
         local net =
            testTiledFFT(problemsToRun[i], nn.SpatialConvolutionFFTTiledSync)
         reportAndFree(net)
      end
      if testAsync then
         local net =
            testTiledFFT(problemsToRun[i], nn.SpatialConvolutionFFTTiledAsync)
         reportAndFree(net)
      end
      if testIterated then
         local net = testTiledFFT(
            problemsToRun[i], nn.SpatialConvolutionFFTTiledIterated)
         reportAndFree(net)
      end
   end
   for step = 1, numTests do
      if testSync then
         local net = testTiledFFT({}, nn.SpatialConvolutionFFTTiledSync)
         reportAndFree(net)
      end
      if testAsync then
         local net = testTiledFFT({}, nn.SpatialConvolutionFFTTiledAsync)
         reportAndFree(net)
      end
      if testIterated then
         local net = testTiledFFT({}, nn.SpatialConvolutionFFTTiledIterated)
         reportAndFree(net)
      end
  end
end

g_mytester = torch.Tester()
g_mytester:add(fb_test)
g_mytester:run()

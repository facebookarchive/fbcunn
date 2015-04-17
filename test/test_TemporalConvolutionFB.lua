-- Copyright 2004-present Facebook. All Rights Reserved.

require 'math'
require 'cunn'
require 'cutorch'
require 'fbcunn'

require('fb.luaunit')
local torch = require('fbtorch')

torch.setnumthreads(6)
torch.setdefaulttensortype('torch.FloatTensor')

local mytester = torch.Tester()

local precision = 0.00002

local TestTemporalConvolutionFB = {}
local printResults = false

local function _initDeterministic(tensor)
   local i = 1
   tensor:apply(function()
      i = i + 1
      return (i - 1)
      end)
end

local function testLoop(problemSize)
   local batchSize = problemSize[1] or 4 * torch.random(8)
   local nInputFrameSize = problemSize[2] or torch.random(1023) + 1
   local nOutputFrameSize = problemSize[3] or torch.random(1023) + 1
   local kW = problemSize[4] or torch.random(15) + 2
   local dW = problemSize[5] or math.min(kW - 2, torch.random(5)) + 1
   -- iW must be a multiple of kW
   local iW = problemSize[6] or kW * math.max(kW, torch.random(30) + 4 - kW + 1)

   print('Running batchSize=', batchSize,
         ' inputFrameSize=', nInputFrameSize,
         ' outputFrameSize=', nOutputFrameSize,
         ' kW=', kW,
         ' dW=', dW,
         ' iW=', iW);

   local net  =
      nn.TemporalConvolution  (nInputFrameSize, nOutputFrameSize, kW, dW)
   local net2 =
      nn.TemporalConvolutionFB(nInputFrameSize, nOutputFrameSize, kW, dW)

   net2.weight:copy(net.weight)
   net2.bias:copy(net.bias)
   local input = torch.Tensor(batchSize, iW, nInputFrameSize):normal()
   local inputCuda = input:clone():cuda()

   net:cuda()
   local output = net:updateOutput(inputCuda):clone():float()
   net2:cuda()
   local output2 = net2:updateOutput(inputCuda):clone():float()

   if printResults then
      print('input 1\n', inputCuda:clone():float())
      print('input 2\n', inputCuda:clone():float())
      print('weight 1\n', net.weight)
      print('weight 2\n', net2.weight)
      print('bias 1\n', net.bias)
      print('bias 2\n', net2.bias)
      print('output old\n', output)
      print(' VS ')
      print('output new\n', output2)
   end
   local norm = math.sqrt(output:dot(output) + 1e-8)
   mytester:assertle(output:dist(output2) / norm,
     precision, 'error on output: ')

   -- UpdateGradInput
   local gradOutput = torch.Tensor(output:size()):normal()
   local gradOutputCuda = gradOutput:cuda()
   local gradInput = net:updateGradInput(
      inputCuda:clone(), gradOutputCuda:clone()):float():clone()
   local gradInput2 = net2:updateGradInput(
      inputCuda:clone(), gradOutputCuda:clone()):float():clone()

   if printResults then
      print('gradOutput old\n', gradOutputCuda:clone():float())
      print('weight 1\n', net.weight)
      print('weight 2\n', net2.weight)
      print('gradInput 1\n', gradInput)
      print('gradInput 2\n', gradInput2)
   end
   local norm = math.sqrt(gradInput:dot(gradInput) + 1e-8)
   mytester:assertle(gradInput:dist(gradInput2) / norm,
     precision, 'error on output: ')

   -- AccGradParameters
   local scale = torch.random(100) / 100
   local gradOutputCuda = gradOutput:cuda()
   net.gradWeight:zero()
   net.gradBias:zero()
   net:accGradParameters(inputCuda:clone(), gradOutputCuda:clone(), scale)

   if printResults then
      print('input\n', inputCuda:clone():float())
      print('gradOutput\n', gradOutputCuda:clone():float())
      print('gradWeight 1\n', net.gradWeight)
      print('gradWeight 2\n', net2.gradWeight)
   end

   net2.gradWeight:zero()
   net2.gradBias:zero()
   net2.weight:copy(net.weight)
   net2.bias:copy(net.bias)
   net2:accGradParameters(inputCuda:clone(), gradOutputCuda:clone(), scale)

   local gradWeight = net.gradWeight:clone():float()
   local gradWeight2 = net2.gradWeight:clone():float():resizeAs(gradWeight)

   if printResults then
      print('input\n', inputCuda:clone():float())
      print('gradOutput\n', gradOutputCuda:clone():float())
      print('gradWeight 1\n', gradWeight)
      print('gradWeight 2\n', gradWeight2)
   end
   local norm = math.sqrt(gradWeight:dot(gradWeight) + 1e-8)
   mytester:assertle(gradWeight:dist(gradWeight2) / norm,
     precision, 'error on gradWeight')

   if printResults then
      print('gradOutput\n', gradOutputCuda:clone():float())
      print('scale ', scale)
      print('gradBias 1 \n', net.gradBias)
      print('gradBias 2\n', net2.gradBias)
   end
   local norm = math.sqrt(net.gradBias:dot(net.gradBias) + 1e-8)
   mytester:assertle(net.gradBias:dist(net2.gradBias) / norm,
     precision, 'error on gradBias')

   collectgarbage()
end


-- batchSize, iW, oW, kW, dW, iH
local problemSize = {
   {1, 3, 4, 1, 1, 1},
   {1, 3, 4, 1, 1, 2},
   {1, 3, 5, 2, 1, 4},
   {1, 3, 4, 1, 2, 1},
   {1, 3, 4, 1, 2, 2},
   {1, 3, 5, 2, 2, 4},
   {3, 5, 6, 3, 1, 4},
   {14, 5, 6, 3, 1, 4},
   {15, 5, 6, 3, 1, 4},
   {16, 16, 16, 4, 2, 16},
   {64, 512, 512, 2, 2, 32},
   {32, 512, 512, 4, 2, 32},
}

function TestTemporalConvolutionFB.test()
    for _, problem in ipairs(problemSize) do
        testLoop(problem)
    end
    for _=1,5 do
        testLoop({})
    end
end

mytester:add(TestTemporalConvolutionFB)
mytester:run()

-- Copyright 2004-present Facebook. All Rights Reserved.

require 'cunn'
require 'fb.luaunit'
require 'fbtorch'
require 'fbcunn'

local tester = torch.Tester()

local kTolerance = 1e-4

local LookupTableTest = {}

local function nonBatchTest(elements, nInput, nOutput, features_in_dim_2)
   local cpu_emb = nn.LookupTableGPU(nInput, nOutput, features_in_dim_2)
   cpu_emb:reset()
   local gpu_emb = cpu_emb:clone():cuda()
   local cpu_w, cpu_dw = cpu_emb:getParameters()
   local gpu_w, gpu_dw = gpu_emb:getParameters()
   local cpu_input = torch.Tensor(elements)

   for i = 1, elements do
      cpu_input[i] = torch.random(nInput)
   end

   local gpu_input = cpu_input:cuda()
   local cpu_output = cpu_emb:forward(cpu_input)
   local gpu_output = gpu_emb:forward(gpu_input)
   tester:assertlt((cpu_output - gpu_output:double()):abs():max(), kTolerance)

   local cpu_gradOutput = cpu_output:clone():normal()
   local gpu_gradOutput = cpu_gradOutput:cuda()
   cpu_dw:zero()
   gpu_dw:zero()
   cpu_emb:accGradParameters(cpu_input, cpu_gradOutput)
   gpu_emb:accGradParameters(gpu_input, gpu_gradOutput)

   tester:assertlt((cpu_dw - gpu_dw:double()):abs():max(), kTolerance)

   cpu_dw:normal()
   gpu_dw:copy(cpu_dw)
   cpu_emb:accGradParameters(cpu_input, cpu_gradOutput)
   gpu_emb:accGradParameters(gpu_input, gpu_gradOutput)
   tester:assertlt((cpu_dw - gpu_dw:double()):abs():max(), kTolerance)
end

local function batchTest(batch_size, elements, nInput, nOutput,
                         features_in_dim_2)
   local cpu_emb = nn.LookupTableGPU(nInput, nOutput, features_in_dim_2)
   cpu_emb:reset()
   local gpu_emb = cpu_emb:clone():cuda()
   local cpu_w, cpu_dw = cpu_emb:getParameters()
   local gpu_w, gpu_dw = gpu_emb:getParameters()
   local cpu_input = torch.Tensor(batch_size, elements)

   for i = 1, batch_size do
      for j = 1, elements do
         cpu_input[i][j] = torch.random(nInput)
      end
   end

   local gpu_input = cpu_input:cuda()
   local cpu_output = cpu_emb:forward(cpu_input)
   local gpu_output = gpu_emb:forward(gpu_input)
   tester:assertlt((cpu_output - gpu_output:double()):abs():max(), kTolerance)

   local cpu_gradOutput = cpu_output:clone():normal()
   local gpu_gradOutput = cpu_gradOutput:cuda()
   cpu_dw:zero()
   gpu_dw:zero()
   cpu_emb:accGradParameters(cpu_input, cpu_gradOutput)
   gpu_emb:accGradParameters(gpu_input, gpu_gradOutput)
   tester:assertlt((cpu_dw - gpu_dw:double()):abs():max(), kTolerance)

   cpu_dw:normal()
   gpu_dw:copy(cpu_dw)
   cpu_emb:accGradParameters(cpu_input, cpu_gradOutput)
   gpu_emb:accGradParameters(gpu_input, gpu_gradOutput)
   tester:assertlt((cpu_dw - gpu_dw:double()):abs():max(), kTolerance)
end

local function FeatureDim2Test(batch_size, elements, nInput, nOutput)
   local embF3 = nn.LookupTableGPU(nInput, nOutput)
   local embF2 = nn.LookupTableGPU(nInput, nOutput, true)
   local _, dwF3 = embF3:getParameters()
   local _, dwF2 = embF2:getParameters()
   embF3:reset()
   embF2.weight:copy(embF3.weight)

   local input
   if batch_size then
      input = torch.Tensor(batch_size, elements)
      for i = 1, batch_size do
         for j = 1, elements do
            input[i][j] = torch.random(nInput)
         end
      end
   else
      input = torch.Tensor(elements)
      for j = 1, elements do
         input[j] = torch.random(nInput)
      end
   end

   local outputF2 = embF2:forward(input)
   local outputF3 = embF3:forward(input)
   if batch_size then
      tester:assertlt((outputF2:transpose(2, 3) - outputF3):abs():max(),
                                                           kTolerance)
   else
      tester:assertlt((outputF2 - outputF3):abs():max(), kTolerance)
   end

   local gradOutputF2 = outputF2:clone():add(42) -- just to make it != output
   local gradOutputF3 = outputF3:clone():add(42)
   dwF2:zero()
   dwF3:zero()
   embF2:accGradParameters(input, gradOutputF2)
   embF3:accGradParameters(input, gradOutputF3)
   tester:assertlt((dwF2 - dwF3):abs():max(), kTolerance)
end

function LookupTableTest.nonBatchTest()
   for i = 1, 10 do
      local elements = torch.random(30)
      local nInput = torch.random(300)
      local nOutput = torch.random(200)
      nonBatchTest(elements, nInput, nOutput)
      nonBatchTest(1, nInput, nOutput)
      nonBatchTest(elements, 1, nOutput)
      nonBatchTest(elements, nInput, 1)
      nonBatchTest(elements, nInput, nOutput, true)
      nonBatchTest(1, nInput, nOutput, true)
      nonBatchTest(elements, 1, nOutput, true)
      nonBatchTest(elements, nInput, 1, true)
   end
end

function LookupTableTest.batchTest()
   for i = 1, 10 do
      local batch_size = torch.random(5)
      local elements = torch.random(30)
      local nInput = torch.random(300)
      local nOutput = torch.random(200)
      batchTest(batch_size, elements, nInput, nOutput)
      batchTest(batch_size, 1, nInput, nOutput)
      batchTest(batch_size, elements, 1, nOutput)
      batchTest(batch_size, elements, nInput, 1)
      batchTest(batch_size, elements, nInput, nOutput, true)
      batchTest(batch_size, 1, nInput, nOutput, true)
      batchTest(batch_size, elements, 1, nOutput, true)
      batchTest(batch_size, elements, nInput, 1, true)
   end
end

function LookupTableTest.featureDim2Test()
   for i = 1, 10 do
      local batch_size = torch.random(5)
      local elements = torch.random(30)
      local nInput = torch.random(300)
      local nOutput = torch.random(200)
      FeatureDim2Test(batch_size, elements, nInput, nOutput)
      FeatureDim2Test(nil, elements, nInput, nOutput)
   end
end

local function exactEquality(a, b)
   for i = 1, a:size(1) do
      for j = 1, a:size(2) do
         if (a[i][j] ~= b[i][j]) then
            print('Diff ' .. string.format('%.9f', a[i][j]) .. ' ' ..
                     string.format('%.9f', b[i][j]))
         end
      end
   end
end

function LookupTableTest.determinismTest()
   -- Test for determistic outputs in accGradParameters
   local batch_size = 30
   local elements = 60
   local classes = 50
   local dim = 50

   local m = nn.LookupTableGPU(classes, dim):cuda()
   local input = torch.rand(batch_size, elements):cuda():mul(classes):ceil()
   local gradOutput = torch.randn(batch_size, elements, dim):cuda()

   -- Create baseline
   m:forward(input)
   m.gradWeight:zero()
   m:backward(input, gradOutput)

   local baseGradWeight = m.gradWeight:clone():float()

   for i = 1, 10 do
      m.gradWeight:zero()
      m:backward(input, gradOutput)

      -- Compare new updated gradient with baseline
      exactEquality(baseGradWeight, m.gradWeight:float());
   end
end

tester:add(LookupTableTest)
tester:run()

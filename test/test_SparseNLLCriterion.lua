-- Copyright 2004-present Facebook. All Rights Reserved.
-- Author: Michael Mathieu <myrhev@fb.com>

require 'fb.luaunit'
require 'cutorch'
require 'nn'
require 'fbcunn'
require 'fbnn'

local test_repeats = 5

local function test_finite_diff_gradInput(model, input, target)
    local eps = 1e-3
    local output = model:updateOutput(input, target)
    local gradInput = model:updateGradInput(input, target):clone()

    local gradInput2 = torch.Tensor(input:size())
    local outputP = torch.Tensor(output:size())
    local outputM = torch.Tensor(output:size())
    if input:dim() == 1 then
        for i = 1,input:size(1) do
            input[i] = input[i] + eps
            outputP:copy(model:updateOutput(input, target))
            input[i] = input[i] - 2*eps
            outputM:copy(model:updateOutput(input, target))
            input[i] = input[i] + eps
            gradInput2[i] = outputP:add(-1, outputM):div(2*eps)
        end
    else
        assert(input:dim() == 2)
        for i = 1,input:size(1) do
            for j = 1,input:size(2) do
                input[i][j] = input[i][j] + eps
                outputP:copy(model:updateOutput(input, target))
                input[i][j] = input[i][j] - 2*eps
                outputM:copy(model:updateOutput(input, target))
                input[i][j] = input[i][j] + eps
                gradInput2[i][j] = outputP:add(-1, outputM):div(2*eps)
            end
        end
    end
    gradInput2 = gradInput2:type(input:type())
    return gradInput:add(-1, gradInput2):abs():max()
end

local function test_sparseNLL(K, n_classes, batch_size, cuda)
   local crit = nn.SparseNLLCriterion(K)
   local input = torch.randn(batch_size, n_classes)
   local targetP = torch.randn(batch_size, K):abs()
   local targetIdx = torch.LongTensor(batch_size, K)
   if cuda then
      crit = crit:cuda()
      input = input:cuda()
      targetP = targetP:cuda()
      targetIdx = torch.CudaTensor(targetIdx:size()):copy(targetIdx)
   end
   for i = 1, batch_size do
      targetP[i]:div(targetP[i]:sum())
      local p = torch.randperm(n_classes)
      targetIdx[i]:copy(p[{{1,K}}])
   end
   -- fprop
   local output_test = 0
   for i = 1, batch_size do
      for j = 1, K do
         output_test = output_test - input[i][targetIdx[i][j] ] * targetP[i][j]
      end
   end
   output_test = output_test / batch_size
   local fprop_err =
      math.abs(output_test - crit:forward(input, {targetP, targetIdx})[1])

   --bprop
   local bprop_err =
      test_finite_diff_gradInput(crit, input, {targetP, targetIdx})

   return fprop_err, bprop_err
end

function testSparseNLLCriterion()
   for k = 1, test_repeats do
      local n_classes = torch.random(1000)
      local K = torch.random(n_classes)
      local batch_size = torch.random(100)
      local err1, err2 = test_sparseNLL(K, n_classes, batch_size, false)
      assertTrue(err1 < 1e-3)
      assertTrue(err2 < 1e-3)
      local err1, err2 = test_sparseNLL(K, n_classes, batch_size, true)
      assertTrue(err1 < 1e-3)
      assertTrue(err2 < 1e-3)
   end
end

LuaUnit:main()

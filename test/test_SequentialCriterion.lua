-- Copyright 2004-present Facebook. All Rights Reserved.
-- Author: Michael Mathieu <myrhev@fb.com>

require 'fb.luaunit'
require 'nn'
require 'fbcunn'
require 'fbnn'

local test_repeats = 100

local function testSequentialCriterion_run(input_size, n_classes,
                                           module, crit, targettype)
   module = module:clone()
   crit = crit:clone()
   local modcrit = nn.SequentialCriterion(module:clone(), crit:clone())
   targettype = targettype or torch.Tensor():type()

   local batch_size = torch.random(100)
   local input = torch.rand(batch_size, input_size)
   local target =
      torch.rand(batch_size):mul(n_classes):add(1):floor():type(targettype)

   local output1 = modcrit:forward(input, target)
   local z2 = module:forward(input)
   local output2 = crit:forward(z2, target)
   assertTrue(math.abs(output1-output2) < 1e-5)

   local gradInput1 = modcrit:updateGradInput(input, target)
   local derr_do2 = crit:updateGradInput(z2, target)
   local gradInput2 = module:updateGradInput(input, derr_do2)
   assertTrue(gradInput1:clone():add(-1, gradInput2):abs():max() < 1e-5)

   modcrit:zeroGradParameters()
   module:zeroGradParameters()
   if crit.zeroGradParameters then
      crit:zeroGradParameters()
   end
   modcrit:accGradParameters(input, target)
   if crit.accGradParameters then
      crit:accGradParameters(z2, target)
   end
   module:accGradParameters(input, derr_do2)
   modcrit:updateParameters(1)
   if crit.updateParameters then
      crit:updateParameters(1)
   end
   module:updateParameters(1)
   local output1 = modcrit:forward(input, target)
   local z2 = module:forward(input)
   local output2 = crit:forward(z2, target)
   assertTrue(math.abs(output1-output2) < 1e-5)
end

local function make_HSM(n_clusters, n_class, input_size)
   local mapping = {}
   local n_class_in_cluster = {}
   for i = 1, n_class do
      local cluster = torch.random(n_clusters)
      n_class_in_cluster[cluster] = n_class_in_cluster[cluster] or 0
      n_class_in_cluster[cluster] = n_class_in_cluster[cluster] + 1
      mapping[i] = {cluster, n_class_in_cluster[cluster]}
   end
   for i = 1,n_clusters do
      if n_class_in_cluster[i] == nil then
         n_class_in_cluster[i] = 1
         mapping[1+#mapping] = {i, 1}
         n_class = n_class + 1
      end
   end
   return nn.HSM(mapping, input_size)
end

function testSequentialCriterion()
   for i = 1, test_repeats do
      -- try with NLL
      local input_size = torch.random(200)
      local n_classes = torch.random(200)
      local module = nn.Linear(input_size, n_classes)
      local crit = nn.ClassNLLCriterion()
      testSequentialCriterion_run(input_size, n_classes, module, crit)

      -- try with HSM
      local input1_size = torch.random(200)
      local input2_size = torch.random(200)
      local n_classes = torch.random(200)
      local module = nn.Sequential()
      module:add(nn.Linear(input1_size, input2_size))
      module:add(nn.Threshold())
      local crit = make_HSM(20, n_classes, input2_size)
      testSequentialCriterion_run(input1_size, n_classes, module,
                                  crit, 'torch.LongTensor')
   end
end

LuaUnit:main()

-- Copyright 2004-present Facebook. All Rights Reserved.

require('nn')

local WeightedLookupTable, parent =
   torch.class('nn.WeightedLookupTable', 'nn.Module')

WeightedLookupTable.__version = 1

function WeightedLookupTable:__init(nIndex, ...)
   parent.__init(self)

   local arg = {...}

   if select('#', ...) == 1 and type(arg[1]) ~= "number" then
      local size = arg[1]
      self.size = torch.LongStorage(#size + 1)
      for i=1,#size do
         self.size[i+1] = size[i]
      end
   else
      self.size = torch.LongStorage(select('#', ...)+1)
      for i=1,select('#',...) do
         self.size[i+1] = arg[i]
      end
   end
   self.size[1] = nIndex

   self.weight = torch.Tensor(self.size)
   self.gradWeight = torch.Tensor(self.size):zero()
   self.input_weights = {}

   self:reset()
end

function WeightedLookupTable:reset(stdv)
   stdv = stdv or 1
   if nn.oldSeed then
      self.weight:apply(function()
         return torch.normal(0, stdv)
      end)
   else
      self.weight:normal(0, stdv)
   end
end

--[[
Parameters:
* `Input` should be an n x 2 tensor where the first column is dictionary indexes
   and the second column is weights.
]]
function WeightedLookupTable:updateOutput(input)
   local output_size = torch.LongStorage(self.size:size()):copy(self.size)
   output_size[1] = input:size(1)
   self.output:resize(output_size):zero()

   for i=1, input:size(1) do
      self.output[i]:add(input[i][2], self.weight[input[i][1]])
   end

   return self.output
end

function WeightedLookupTable:zeroGradParameters()
   for k,_ in pairs(self.input_weights) do
      self.gradWeight:select(1, k):zero()
   end
   self.input_weights = {}
end

function WeightedLookupTable:accGradParameters(input, gradOutput, scale)
   for i=1,input:size(1) do
      local k = input[i][1]
      self.input_weights[k] = true
      self.gradWeight[k]:add(scale * input[i][2], gradOutput[i])
   end
end

function WeightedLookupTable:accUpdateGradParameters(input, gradOutput, lr)
   for i=1,input:size(1) do
      self.weight[input[i][1]]:add(-lr * input[i][2], gradOutput[i])
   end
end

function WeightedLookupTable:updateParameters(learningRate)
   for k,_ in pairs(self.input_weights) do
      self.weight[k]:add(-learningRate, self.gradWeight[k])

   end
end

-- we do not need to accumulate parameters when sharing
WeightedLookupTable.sharedAccUpdateGradParameters = WeightedLookupTable.accUpdateGradParameters

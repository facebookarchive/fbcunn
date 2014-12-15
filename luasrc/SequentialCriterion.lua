-- Copyright 2004-present Facebook. All Rights Reserved.
-- Author: Michael Mathieu <myrhev@fb.com>

--[[
Combines a module and a criterion.

It is mainly thought for preprocessing, but trainable parameters
can be used if needed
]]
local SequentialCriterion, parent =
   torch.class('nn.SequentialCriterion', 'nn.Criterion')

function SequentialCriterion:__init(module, criterion)
   parent.__init(self)
   self.module = module
   self.criterion = criterion
end

function SequentialCriterion:parameters()
   local params, gradParams = self.module:parameters_one()
   if self.criterion.parameters then
      local cparams, cgradParams = self.criterion:parameters_one()
      for i = 1, #cparams do
         params[1+#params] = cparams[i]
         gradParams[1+#gradParams] = cgradParams[i]
      end
   end
   return params, gradParams
end

function SequentialCriterion:getParameters()
   return nn.Module.getParameters(self)
end

function SequentialCriterion:updateOutput(input, target)
   self.module:updateOutput(input)
   self.output = self.criterion:updateOutput(self.module.output, target)
   return self.output
end

function SequentialCriterion:updateGradInput(input, target)
   local derr_do = self.criterion:updateGradInput(self.module.output, target)
   self.gradInput = self.module:updateGradInput(input, derr_do)
   return self.gradInput
end

function SequentialCriterion:accGradParameters(input, target, scale)
   if self.criterion.accGradParameters ~= nil then
      self.criterion:accGradParameters(self.module.output, target, scale)
   end
   self.module:accGradParameters(input, self.criterion.gradInput, scale)
end

function SequentialCriterion:accUpdateGradParameters(input, target, scale)
   if self.criterion.accUpdateGradParameters ~= nil then
      self.criterion:accUpdateGradParameters(self.module.output, target, scale)
   end
   self.module:accUpdateGradParameters(input, self.criterion.gradInput, scale)
end

function SequentialCriterion:updateParameters(learning_rate)
   self.module:updateParameters(learning_rate)
   if self.criterion.updateParameters then
      self.criterion:updateParameters(learning_rate)
   end
end

function SequentialCriterion:zeroGradParameters()
   if self.criterion.zeroGradParameters then
      self.criterion:zeroGradParameters()
   end
   self.module:zeroGradParameters()
end

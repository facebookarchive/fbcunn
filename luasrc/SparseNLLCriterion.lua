-- Copyright 2004-present Facebook. All Rights Reserved.
-- Author: Michael Mathieu <myrhev@fb.com>

require 'math'
require 'nn'

-- Sparse ClassNLL criterion
local SparseNLLCriterion, parent =
    torch.class('nn.SparseNLLCriterion', 'nn.Criterion')

--[[
Parameters:
* `K` : number of non-zero elements of the target
* `do`_target_check : checks whether the target is a
   probability vector (default true)
* `sizeAverage` : divides the error by the size of the minibatch
]]
function SparseNLLCriterion:__init(K)
   parent.__init(self)
   self.K = K
   self.do_target_check = true
   self.sizeAverage = true
   self.output = torch.Tensor(1)
   self.gradInput = torch.Tensor()
   self.tmp_buffer = torch.Tensor()
end

--[[
`target` should be a table containing two tensors :

```
target = {targetP, targetIdx}
```

where `targetP` are the probabilities associated to the indices `targetIdx`
we assume `targetIdx` doesn't have twice the same number in the same sample.
]]
function SparseNLLCriterion:updateOutput(input, target)
   -- minibatches
   if input:dim() == 1 then
      input = input:reshape(1, input:size(1))
      target[1] = target[1]:reshape(1, target[1]:size(1))
      target[2] = target[2]:reshape(1, target[2]:size(1))
   else
      assert(input:dim() == 2)
   end

   -- tests if the target sums to 1
   if self.do_target_check then
      self.tmp_buffer:resize(input:size(1), 1)
      self.tmp_buffer:sum(target[1], 2)
      if self.tmp_buffer:add(-1):abs():max() > 1e-3 then
         error('SparseNLLCriterion : input is not a probability vector \
(you can disable this error by setting do_target_check to false)')
      end
   end

   -- compute the output
   input.nn.SparseNLLCriterion_updateOutput(self, input, target[1], target[2])

   -- sizeAverage ?
   if self.sizeAverage then
      self.output:div(input:size(1))
   end

   return self.output
end

function SparseNLLCriterion:updateGradInput(input, target)
   self.gradInput:resizeAs(input)
   input.nn.SparseNLLCriterion_updateGradInput(self, target[1], target[2])
   if self.sizeAverage then
      self.gradInput:div(input:size(1))
   end
   return self.gradInput
end

-- Copyright 2004-present Facebook. All Rights Reserved.
-- Author: Michael Mathieu <myrhev@fb.com>

require 'cunn'

--[[
Fast lookup table, supporting both CPU and GPU modes.
]]
local LookupTableGPU, parent = torch.class('nn.LookupTableGPU', 'nn.Module')


--[[
If `featuresInDim2` is `true`, an input of dimension `batchSize` ${\times}$ `N` will produce an output of size `batchSize` ${\times}$ `nOutput`
${\times}$ `N`. If it is set to `false` (default) it will produce an output
of size `batchSize` ${\times}$ `N` ${\times}$ `nOutput`.
]]
function LookupTableGPU:__init(nInput, nOutput, featuresInDim2)
   parent:__init(self)
   self.nInput = nInput
   self.nOutput = nOutput
   self.featuresInDim2 = featuresInDim2 or false
   -- Careful : weight is transposed from nn.Linear
   self.weight = torch.Tensor(nInput, nOutput)
   self.gradWeight = torch.Tensor(nInput, nOutput)
   self.output = torch.Tensor()

   self:reset()
end

function LookupTableGPU:reset(stdv)
   stdv = stdv or 1
   self.weight:normal(stdv)
end

function LookupTableGPU:parameters()
    return {self.weight}, {self.gradWeight}
end

-- input should be a 1d (size N) or 2d (size batchSize x N)
-- tensor of byte or long on CPU, cudaTensor on GPU.
-- It contains the indices of the lookup.
function LookupTableGPU:updateOutput(input)
   if input:dim() == 2 then
      if self.featuresInDim2 then
         self.output:resize(input:size(1), self.nOutput, input:size(2))
      else
         self.output:resize(input:size(1), input:size(2), self.nOutput)
      end
   else
      self.output:resize(input:size(1), self.nOutput)
   end

   if input:type() == 'torch.CudaTensor' then
      input.nn.LookupTableGPU_updateOutput(input, self.weight, self.output,
                                           self.featuresInDim2)
   else
      if input:dim() == 2 then
         -- batch mode
         local this_output
         for batch = 1, input:size(1) do
            for i = 1, input:size(2) do
               if self.featuresInDim2 then
                  this_output = self.output[{batch, {}, i}]
               else
                  this_output = self.output[{batch, i}]
               end
               if self.unk_index and (input[batch][i] == self.unk_index) then
                  this_output:zero()
               else
                  this_output:copy(self.weight[input[batch][i]])
               end
            end
         end
      else
         -- non-batch mode
         if input:size(1) == 1 then
            if self.unk_index and (input[1] == self.unk_index) then
               self.zeros = self.zeros or torch.zeros(1, self.nOutput)
               self.output = self.zeros
            else
               self.output = self.weight[input[1]]:reshape(1, self.nOutput)
            end
         else
            self.output:resize(input:size(1), self.nOutput)
            for i = 1,input:size(1) do
               if self.unk_index and (input[i] == self.unk_index) then
                  self.output[i]:zero()
               else
                  self.output[i]:copy(self.weight[input[i]])
               end
            end
         end
      end
   end

   return self.output
end

function LookupTableGPU:updateGradInput(input, gradOutput)
    --print("Should not be used") --TODO
end

function LookupTableGPU:accGradParameters(input, gradOutput, scale)
    scale = scale or 1
    if input:type() == 'torch.CudaTensor' then
        input.nn.LookupTableGPU_accGradParameters(input, gradOutput,
                                                  self.gradWeight, scale,
                                                  self.featuresInDim2)
    else
       if input:dim() == 2 then
          -- batch mode
          for batch = 1, input:size(1) do
             for i = 1, input:size(2) do
                if (self.unk_index == nil) or
                (input[batch][i] ~= self.unk_index) then
                   if self.featuresInDim2 then
                      self.gradWeight[input[batch][i]]
                      :add(scale, gradOutput[{batch, {}, i}])
                   else
                      self.gradWeight[input[batch][i]]
                      :add(scale, gradOutput[batch][i])
                   end
                end
             end
          end
       else
          -- non-batch mode
          for i = 1,input:size(1) do
             if (self.unk_index == nil) or
             (input[i] ~= self.unk_index) then
                self.gradWeight[input[i] ]:add(scale, gradOutput[i])
             end
          end
       end
    end
end

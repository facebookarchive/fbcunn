-- Copyright 2004-present Facebook. All Rights Reserved.

require 'nn'

local TemporalConvolutionFB, parent =
   torch.class('nn.TemporalConvolutionFB', 'nn.Module')

function TemporalConvolutionFB:__init(inputFrameSize, outputFrameSize, kW, dW)
   parent.__init(self)

   dW = dW or 1

   self.inputFrameSize = inputFrameSize
   self.outputFrameSize = outputFrameSize
   self.kW = kW
   self.dW = dW

   self.weight = torch.Tensor(outputFrameSize, kW, inputFrameSize)
   self.bias = torch.Tensor(outputFrameSize)
   self.gradWeight = torch.Tensor(outputFrameSize, kW, inputFrameSize)
   self.gradBias = torch.Tensor(outputFrameSize)

   self:reset()
end

function TemporalConvolutionFB:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1/math.sqrt(self.kW*self.inputFrameSize)
   end
   if nn.oldSeed then
      self.weight:apply(function()
         return torch.uniform(-stdv, stdv)
      end)
      self.bias:apply(function()
         return torch.uniform(-stdv, stdv)
      end)
   else
      self.weight:uniform(-stdv, stdv)
      self.bias:uniform(-stdv, stdv)
   end
end

function TemporalConvolutionFB:updateOutput(input)
   input.nn.TemporalConvolutionFB_updateOutput(self, input)
   return self.output
end

function TemporalConvolutionFB:updateGradInput(input, gradOutput)
   if self.gradInput then
      return input.nn.TemporalConvolutionFB_updateGradInput(
         self, input, gradOutput)
   end
end

function TemporalConvolutionFB:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   input.nn.TemporalConvolutionFB_accGradParameters(
      self, input, gradOutput, scale)
end

-- we do not need to accumulate parameters when sharing
TemporalConvolutionFB.sharedAccUpdateGradParameters =
  TemporalConvolutionFB.accUpdateGradParameters

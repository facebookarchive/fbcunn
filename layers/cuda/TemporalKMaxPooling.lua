-- Copyright 2004-present Facebook. All Rights Reserved.

require 'cutorch'
require 'nn'

local TemporalKMaxPooling, parent =
    torch.class('nn.TemporalKMaxPooling', 'nn.Module')

function TemporalKMaxPooling:__init(k, k_dynamic)
    parent.__init(self)

    self.k = k
    self.k_dynamic = k_dynamic or -1

    self.output = torch.CudaTensor()
    self.gradInput = torch.CudaTensor()
    self.indices = torch.CudaTensor()
end

function TemporalKMaxPooling:updateOutput(input)
    input = input:contiguous()
    input.nn.TemporalKMaxPooling_updateOutput(self, input)
    return self.output
end

function TemporalKMaxPooling:updateGradInput(input, gradOutput)
    input = input:contiguous()
    gradOutput = gradOutput:contiguous()

    input.nn.TemporalKMaxPooling_updateGradInput(self, input, gradOutput)
    return self.gradInput
end

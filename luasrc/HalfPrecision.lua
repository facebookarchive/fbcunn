-- Copyright 2004-present Facebook. All Rights Reserved.

local libhalfprec = require('libhalfprec')

local function truncate(floats)
    return libhalfprec.toFloatCUDA(libhalfprec.toHalfCUDA(floats))
end

local HalfPrecision, parent =
    torch.class('nn.HalfPrecision', 'nn.Module')

function HalfPrecision:__init()
    parent.__init(self)
    self.output = torch.CudaTensor()
    self.gradInput = torch.CudaTensor()
end

function HalfPrecision:updateOutput(input)
    input = input:contiguous():cuda()
    self.output = truncate(input)
    self.output:resizeAs(input)
    return self.output
end

function HalfPrecision:updateGradInput(input, gradOutput)
    self.gradInput = truncate(gradOutput)
    self.gradInput:resizeAs(gradOutput)
    return self.gradInput
end

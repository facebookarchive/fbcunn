-- Copyright 2004-present Facebook. All Rights Reserved.

-- TemporalKmaxPooling
-- Input : (bsize x) width x height
-- Output : (bisze x) k_out x height
-- with k_out = max(k_out_prop, inputSeqLen)
-- where k_out_prop = max(k, ceil(k_dynamic*inputSeqLen))

require 'cutorch'
require 'nn'

local TemporalKMaxPooling, parent =
    torch.class('nn.TemporalKMaxPooling', 'nn.Module')

function TemporalKMaxPooling:__init(k, k_dynamic)
    parent.__init(self)

    self.k = k
    if k_dynamic then
        assert(k_dynamic <= 1 and k_dynamic >=0,
        'k_dynamic must be between 0 and 1')
    end
    self.k_dynamic = k_dynamic or -1

    -- k_dynamic is an optional scalar parameter between 0 and 1
    -- that makes k a fraction of the input sequence size.

    -- To follow Kalchbrenner et al's architecture on Dynamic k-Max Pooling:
    -- Use (k = k_top, kDynamic = (L - l)/L), with
    -- L : total number of conv layers,
    -- l : current convolutional layer to which the pooling is applied,
    -- k_top : fixed pooling parameter for the topmost convolutional layer.

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

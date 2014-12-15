-- Copyright 2004-present Facebook. All Rights Reserved.

--[[
Cross-map normalization, see
https://code.google.com/p/cuda-convnet/wiki/LayerParams#Local_response_normalization_layer_(across_maps)

formula:

$${f(u_{f}^{x,y})=\frac{u_{f}^{x,y}}{ (1+\frac{\alpha}{N} \sum_{f'=\max(0,F-\lfloor N/2\rfloor )}^{\min(F,f-\lfloor N/2 \rfloor+N) }(u_{f'}^{x,y})^{2})^{\beta}}}$$

where
* ${F}$ is the number of features, 
* ${N}$ is the neighborhood size (size),
* ${\alpha}$ is the scaling factor (scale),
* ${\beta}$ is the exponent (power)

This layer normalizes values across feature maps (each spatial location
independently). Borders are zero-padded.

Parameters:
* `size`: size of the neighborhood (typical value: 5)
* `scale`: scaling factor (typical value: 0.0001)
* `power`: exponent used (typical value: 0.75)
]]
local CrossMapNormalization, parent =
    torch.class('nn.CrossMapNormalization', 'nn.Module')

function CrossMapNormalization:__init(size, scale, power)
    parent.__init(self)

    self.size = size
    self.scale = scale
    self.power = power
    self.output = torch.Tensor()
    self.gradInput = torch.Tensor()
    -- squaredSum is an intermediate results cache computed
    -- during updateOutput() and used b updateGradInput() to
    -- speedup computation.
    self.squaredSum = torch.Tensor()
end

function CrossMapNormalization:updateOutput(input)
    return input.nn.CrossMapNormalization_updateOutput(self, input)
end

function CrossMapNormalization:updateGradInput(input, gradOutput)
    return input.nn.CrossMapNormalization_updateGradInput(
        self, input, gradOutput)
end

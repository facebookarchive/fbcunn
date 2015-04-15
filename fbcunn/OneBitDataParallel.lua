require('cutorch')
require 'fbnn'
local util = require('fb.util')
local OBSGD = require('fbcunn.OneBitSGD')
local withDevice = cutorch.withDevice

--[[ OneBitDataParallel implements the "1-Bit Stochastic Gradient
Descent and Application to Data-Parallel Distributed Training of
Speech DNNs" paper of Frank Seide, Hao Fu, Jasha Droppo, Gang Li, and
Dong Yu.

The implementation is similar to a vanilla DataParallel module, except we replace the averaging gradient step with a quantize-copy-merge-broadcast procedure.

<http://research.microsoft.com/apps/pubs/?id=230137>
]]
local OneBitDataParallel, parent =
    torch.class('nn.OneBitDataParallel', 'nn.DataParallel')

function OneBitDataParallel:__init(dimension, config)
    parent.__init(self, dimension)
    self.config = {}
    self.config.min_elements = config.min_elements or 20
    self.config.adagrad_learning_rate = config.learningRate or 1.0
    self.config.momentum_rate = config.momentum or 0

    -- Aggregators for each [row][gradient]
    self._aggregators = util.defaultdict(function() return end)
    self.home_device = 1
end


function OneBitDataParallel:_should_run_one_bit_sgd(gradients)
    -- TODO(tulloch) - flesh this test out
    assert(gradients)
    assert(#gradients >= 1)
    return gradients[1]:nDimension() == 2 and
        gradients[1]:nElement() > self.config.min_elements
end

function OneBitDataParallel:_combine_gradients(row_idx, gradients)
    assert(#gradients >= 1)
    if not self:_should_run_one_bit_sgd(gradients) then
        return parent._combine_gradients(self, row_idx, gradients)
    end

    if not self._aggregators[row_idx] then
        local g = gradients[1]
        self._aggregators[row_idx] = OBSGD.OneBitAggregator(
            self.config,
            function() return torch.Tensor():typeAs(g):resizeAs(g):zero() end,
            function(dest, source) return self:gpuSend(dest, source) end,
            self.home_device
        )
        self.home_device = self.home_device % cutorch.getDeviceCount() + 1
    end
    self._aggregators[row_idx]:run(gradients)
end

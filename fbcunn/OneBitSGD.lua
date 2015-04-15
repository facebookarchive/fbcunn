--[[
OneBitSGD contains various utility functions for use in OneBitDataParallel, exported for unit testing purposes.
]]

local M = {}

local _fbd = require('fb.debugger')
local _trace = require('fb.util.trace')

local pl = require('pl.import_into')()
local util = require('fb.util')
local withDevice = cutorch.withDevice

local function isSameSizeAsRegardlessOfType(l, r)
    return l:typeAs(r):isSameSizeAs(r)
end

-- Run Adagrad, returning the updated history and per-coordinate
-- adjusted learning rates.
local function adagrad(gradient, base_learning_rate, adagrad_history, adagrad_gradient, gradient_like)
    assert(gradient:isSameSizeAs(adagrad_history))
    gradient_like:copy(gradient):cmul(gradient)
    adagrad_history:add(gradient_like)
    gradient_like:copy(adagrad_history):sqrt()
    return adagrad_gradient:copy(gradient):cdiv(gradient_like):mul(base_learning_rate)

    --local new_adagrad_history =
    --    adagrad_history:clone():add(gradient:clone():cmul(gradient))
    --local adagrad_denominator = new_adagrad_history:clone():sqrt()

    -- compute
    --    g' <- (g * learning_rate) / sqrt(sum g_i^2})
    --local adagrad_gradient =
    --    gradient:clone():cdiv(adagrad_denominator):mul(base_learning_rate)
    --return adagrad_gradient, new_adagrad_history
end
M.adagrad = adagrad

-- Apply momentum to the gradient, returning a new gradient with
-- momentum applied.
local function momentum(gradient, momentum_rate, momentum_history)
    assert(gradient:isSameSizeAs(momentum_history))
    return momentum_history:mul(momentum_rate):add(gradient)
end
M.momentum = momentum


-- Unquantize the given one-bit quantized gradient with the
-- given unquantization dictionary.
-- Returns the unquantized gradient.
local function unquantize_gradient(quantized_gradient, avg_pos, avg_neg, num_orig_cols, quantizer)
    assert(avg_pos:size()[1] == quantized_gradient:size()[1])
    assert(avg_neg:size()[1] == quantized_gradient:size()[1])

    return quantizer:dequantize(quantized_gradient, avg_pos, avg_neg, num_orig_cols)
end
M.unquantize_gradient = unquantize_gradient

local function quantize_gradient(gradient, quantizer, accumulated)
    assert(gradient:nDimension() == 2)
    assert(gradient:isSameSizeAs(accumulated))
    accumulated:copy(gradient)
    if quantizer.quantization_error then
        assert(quantizer.quantization_error:isSameSizeAs(gradient))
        accumulated:add(quantizer.quantization_error)
    end

    local quantized_gradient, qe, avg_pos, avg_neg =
        quantizer:quantize(accumulated)

    return quantized_gradient, avg_pos, avg_neg
end

M.quantize_gradient = quantize_gradient

--------------------------------------------------------------------------------
-- OneBitAggregator
local OneBitAggregator = pl.class()
M.OneBitAggregator = OneBitAggregator

function OneBitAggregator:_init(config, gradient_tensor_factory, sender_function, home_device)
    self.config = config
    self.gradient_tensor_factory = gradient_tensor_factory
    self.sender_function = sender_function
    self.home_device = home_device
    -- quantizer for each column
    self.remote_quantizer = util.defaultdict(function() return nn.OneBitQuantization() end)

    self.home_quantizer = withDevice(home_device, function() return nn.OneBitQuantization() end)
    self.adagrad_history = withDevice(home_device, function() return gradient_tensor_factory() end)
    self.momentum_history = withDevice(home_device, function() return gradient_tensor_factory() end)
    self.adagrad_gradient = withDevice(home_device, function() return gradient_tensor_factory() end)
    self.gradient_like = withDevice(home_device, function() return gradient_tensor_factory() end)

    self.remote_quantized_gradient = util.defaultdict(function() return end)
    self.remote_avg_pos = util.defaultdict(function() return end)
    self.remote_avg_neg = util.defaultdict(function() return end)
    self.remote_accumulated = util.defaultdict(function() return end)

    self.home_quantized_gradient = nil
    self.home_avg_pos = nil
    self.home_avg_neg = nil
    self.home_accumulated = nil

    self.gradient_avg = nil

end

function OneBitAggregator:run(gradients)
    assert(isSameSizeAsRegardlessOfType(self.adagrad_history, gradients[1]))
        local average_gradient =
            self:_accumulate_quantized_gradients(gradients)
        local smoothed_gradient =
            self:_smooth_gradient(average_gradient)
        local quantized_gradient, avg_pos, avg_neg = withDevice(self.home_device, function()
            return quantize_gradient(smoothed_gradient, self.home_quantizer, self.home_accumulated)
            end)
        cutorch.synchronize()

        self:_broadcast_gradient(quantized_gradient, avg_pos, avg_neg, gradients)

        cutorch.synchronize()

end

function OneBitAggregator:_smooth_gradient(average_gradient)
    -- TODO(tulloch) there are a huge number of things we can do here.
    -- In the paper, they claim that applying Adagrad to unquantized
    -- gradients performs better than after applying momentum.
    -- This is something that needs to be experimented with.
    -- For now, just implement adagrad-then-momentum.
    return withDevice(
        self.home_device,
        function()
            local adagrad_gradient = adagrad(
                average_gradient,
                self.config.adagrad_learning_rate,
                self.adagrad_history, self.adagrad_gradient, self.gradient_like)
            local momentum_gradient = momentum(
                adagrad_gradient, self.config.momentum_rate, self.momentum_history)
            return momentum_gradient
        end
        )
end

function OneBitAggregator:_broadcast_gradient(
        quantized_gradient, avg_pos, avg_neg, gradients)
    -- Broadcast aggregated gradients to all GPUs and apply.

    local function apply_gradient_remotely(gradient)
        -- Copy quantized gradients to the remote GPU
        local column = gradient:getDevice()

        if not self.remote_quantized_gradient[column] then
            self.remote_quantized_gradient[column] =
                torch.Tensor():typeAs(quantized_gradient):resizeAs(quantized_gradient)
        end
        if not self.remote_avg_pos[column] then
            self.remote_avg_pos[column] =
                torch.Tensor():typeAs(avg_pos):resizeAs(avg_pos)
        end
        if not self.remote_avg_neg[column] then
            self.remote_avg_neg[column] =
                torch.Tensor():typeAs(avg_neg):resizeAs(avg_neg)
        end

        self.sender_function(self.remote_quantized_gradient[column], quantized_gradient)
        self.sender_function(self.remote_avg_pos[column], avg_pos)
        self.sender_function(self.remote_avg_neg[column], avg_neg)

        -- Unquantize the gradients on the remote GPUs
        local unquantized_gradient = unquantize_gradient(
            self.remote_quantized_gradient[column], self.remote_avg_pos[column],
                self.remote_avg_neg[column], gradient:size()[2],
                self.remote_quantizer[gradient:getDevice()])
        -- Update the old gradient in-place
        gradient:copy(unquantized_gradient)
    end
    pl.tablex.foreach(
        gradients,
        function(gradient)
            withDevice(
                gradient:getDevice(),
                function() apply_gradient_remotely(gradient) end
            )
        end
    )
end


function OneBitAggregator:_accumulate_quantized_gradients(gradients)
    -- We do our accumulation work on the home_device.
    local num_orig_cols = gradients[1]:size()[2]

    local quantized_gradient_packs = pl.tablex.pairmap(
        function(column_idx, gradient)
            return self:_gradient_quantization_remote(
                self.home_device, column_idx, gradient, num_orig_cols)
        end,
        gradients
    )
    cutorch.synchronize()

    return self:_average_gradients(quantized_gradient_packs, gradients, num_orig_cols)
end

function OneBitAggregator:_gradient_quantization_remote(
        home_device, column_idx, gradient, num_orig_cols)
    -- Quantize the gradients remotely
    local remote_quantized_gradient, remote_avg_pos, remote_avg_neg
    withDevice(
        gradient:getDevice(),
        function()
            if not self.remote_accumulated[column_idx] then
                self.remote_accumulated[column_idx] =
                    torch.Tensor():typeAs(gradient):resizeAs(gradient)
            end
            remote_quantized_gradient,
            remote_avg_pos,
            remote_avg_neg =
                quantize_gradient(
                    gradient,
                    self.remote_quantizer[column_idx], self.remote_accumulated[column_idx])
        end
    )
    return {remote_quantized_gradient, remote_avg_pos, remote_avg_neg}
end

function OneBitAggregator:_average_gradients(quantized_gradient_packs, gradients, num_orig_cols)
    assert(#quantized_gradient_packs >= 1)


    return withDevice(
        self.home_device,
        function()
            if not self.gradient_avg then
                self.gradient_avg = torch.Tensor():typeAs(gradients[1]):resizeAs(gradients[1]):zero()
            else
                self.gradient_avg:zero()
            end

            for i = 1, #quantized_gradient_packs do
                local quantized_gradient_pack =  quantized_gradient_packs[i]
                local unquantized_gradient
                local remote_quantized_gradient, remote_avg_pos, remote_avg_neg
                remote_quantized_gradient = quantized_gradient_pack[1]
                remote_avg_pos = quantized_gradient_pack[2]
                remote_avg_neg = quantized_gradient_pack[3]

                if not self.home_quantized_gradient then
                    self.home_quantized_gradient =
                    torch.Tensor():typeAs(remote_quantized_gradient):resizeAs(remote_quantized_gradient)
                end
                if not self.home_avg_pos then
                    self.home_avg_pos =
                    torch.Tensor():typeAs(remote_avg_pos):resizeAs(remote_avg_pos)
                end
                if not self.home_avg_neg then
                    self.home_avg_neg =
                    torch.Tensor():typeAs(remote_avg_neg):resizeAs(remote_avg_neg)
                end

                self.sender_function(self.home_avg_pos, remote_avg_pos)
                self.sender_function(self.home_avg_neg, remote_avg_neg)
                self.sender_function(
                    self.home_quantized_gradient, remote_quantized_gradient)
                unquantized_gradient =
                    unquantize_gradient(self.home_quantized_gradient, self.home_avg_pos, self.home_avg_neg,
                        num_orig_cols, self.home_quantizer)
                if not self.home_accumulated then
                    self.home_accumulated =
                    torch.Tensor():typeAs(unquantized_gradient):resizeAs(unquantized_gradient)
                end
                self.gradient_avg:add(unquantized_gradient)
            end
            -- Average the gradients across GPUs
            return self.gradient_avg:div(#gradients)
        end
    )
end

return M

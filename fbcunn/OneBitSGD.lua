local M = {} -- export local functions for unit testing.

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
local function adagrad(gradient, base_learning_rate, adagrad_history)
    assert(gradient:isSameSizeAs(adagrad_history))
    local new_adagrad_history =
        adagrad_history:clone():add(gradient:clone():cmul(gradient))
    local adagrad_denominator = new_adagrad_history:clone():sqrt()

    -- compute
    --    g' <- (g * learning_rate) / sqrt(sum g_i^2})
    local adagrad_gradient =
        gradient:clone():cdiv(adagrad_denominator):mul(base_learning_rate)
    return adagrad_gradient, new_adagrad_history
end
M.adagrad = adagrad

-- Apply momentum to the gradient, returning a new gradient with
-- momentum applied.
local function momentum(gradient, momentum_rate, momentum_history)
    assert(gradient:isSameSizeAs(momentum_history))
    return gradient:clone():add(momentum_history:clone():mul(momentum_rate))
end
M.momentum = momentum

-- Unquantize the given one-bit quantized gradient with the
-- given unquantization dictionary.
-- Returns the unquantized gradient.
local function unquantize_gradient(quantized_gradient, unquantizer)
    assert(unquantizer:size()[2] == 2)
    assert(unquantizer:size()[1] == quantized_gradient:size()[2])

    -- TODO(tulloch) eventually we'll have to coerce this to the
    -- correct type.
    local unquantized_gradient = quantized_gradient:clone()
    for column = 1,quantized_gradient:size()[2] do
        local function l2_loss_unapply(el)
            if el > 0 then
                return unquantizer[{column, 2}]
            else
                return unquantizer[{column, 1}]
            end
        end

        -- apply updates in place
        unquantized_gradient[{{}, column}]:apply(l2_loss_unapply)
    end

    return unquantized_gradient
end
M.unquantize_gradient = unquantize_gradient

local function quantize_gradient(gradient, previous_quantization_error)
    assert(gradient:nDimension() == 2)
    assert(previous_quantization_error:isSameSizeAs(gradient))

    -- The quantized gradients are simply the thresholded variants of
    -- the gradients
    local accumulated = gradient:clone():add(previous_quantization_error)
    local quantized_gradient =
        accumulated:gt(accumulated, 0):typeAs(gradient)

    -- The mapping from each column to the values for {0, 1} for the
    -- quantized gradients.
    local unquantizer =
        torch.Tensor():typeAs(gradient):resize(gradient:size()[2], 2):zero()

    -- Iterate over each *column*
    for column = 1,gradient:size()[2] do
        local positive_sum, positive_count, negative_sum, negative_count =
            0, 0, 0, 0
        -- L2 Loss <=> minimize squared error <=> take average
        -- TODO(tulloch) - L1 Loss?
        local function l2_loss_apply(el)
            if el > 0 then
                positive_sum = positive_sum + el
                positive_count = positive_count + 1
            else
                negative_sum = negative_sum + el
                negative_count = negative_count + 1
            end
        end

        gradient[{{}, column}]:apply(l2_loss_apply)
        -- XXX do we care about NaN's here? I don't think so, since we
        -- won't ever access them at unquantization time.
        unquantizer[{column, 1}] = negative_sum / negative_count
        unquantizer[{column, 2}] = positive_sum / positive_count
    end

    -- Compute the quantization error
    local unquantized_gradient =
        unquantize_gradient(quantized_gradient, unquantizer)
    -- quantization_error = gradient - unquantized_gradient
    local quantization_error =
        gradient:clone():add(unquantized_gradient:clone():mul(-1))
    return quantized_gradient, unquantizer, quantization_error
end
M.quantize_gradient = quantize_gradient

--------------------------------------------------------------------------------
-- OneBitAggregator
local OneBitAggregator = pl.class()
M.OneBitAggregator = OneBitAggregator

function OneBitAggregator:_init(config, gradient_tensor_factory, sender_function)
    self.config = config
    self.gradient_tensor_factory = gradient_tensor_factory
    self.sender_function = sender_function
    -- quantization erorrs for each column
    self.quantization_errors = util.defaultdict(gradient_tensor_factory)

    self.broadcast_quantization_error = gradient_tensor_factory()
    self.adagrad_history = gradient_tensor_factory()
    self.momentum_history = gradient_tensor_factory()
end

function OneBitAggregator:run(gradients)
    assert(isSameSizeAsRegardlessOfType(self.adagrad_history, gradients[1]))
    local average_gradient =
        self:_accumulate_quantized_gradients(gradients)
    local smoothed_gradient =
        self:_smooth_gradient(average_gradient)
    local quantized_gradient, unquantizer, quantization_error =
        quantize_gradient(smoothed_gradient, self.broadcast_quantization_error)
    self.broadcast_quantization_error = quantization_error
    self:_broadcast_gradient(quantized_gradient, unquantizer, gradients)
end

function OneBitAggregator:_smooth_gradient(average_gradient)
    -- TODO(tulloch) there are a huge number of things we can do here.
    -- In the paper, they claim that applying Adagrad to unquantized
    -- gradients performs better than after applying momentum.
    -- This is something that needs to be experimented with.
    -- For now, just implement adagrad-then-momentum.

    local adagrad_gradient, adagrad_history = adagrad(
        average_gradient,
        self.config.adagrad_learning_rate,
        self.adagrad_history)
    local momentum_gradient = momentum(
        adagrad_gradient, self.config.momentum_rate, self.momentum_history)
    self.momentum_history = momentum_gradient
    self.adagrad_history = adagrad_history
    return momentum_gradient
end

function OneBitAggregator:_broadcast_gradient(
        quantized_gradient, unquantizer, gradients)
    -- Broadcast aggregated gradients to all GPUs and apply.

    local function apply_gradient_remotely(gradient)
        -- Copy quantized gradients to the remote GPU
        local remote_quantized_gradient =
            torch.Tensor():typeAs(quantized_gradient):resizeAs(quantized_gradient)
        local remote_unquantizer =
            torch.Tensor():typeAs(unquantizer):resizeAs(unquantizer)
        self.sender_function(remote_quantized_gradient, quantized_gradient)
        self.sender_function(remote_unquantizer, unquantizer)

        -- Unquantize the gradients on the remote GPUs
        local unquantized_gradient = unquantize_gradient(
            remote_quantized_gradient, remote_unquantizer)

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

-- Quantize gradients remotely, copy to home, and unquantize
function OneBitAggregator:_transfer_gradient_to_home(
        home_device, column_idx, gradient)
    -- Quantize the gradients remotely
    local remote_quantized_gradient, remote_unquantizer
    withDevice(
        gradient:getDevice(),
        function()
            remote_quantized_gradient,
            remote_unquantizer,
            self.quantization_errors[column_idx] =
                quantize_gradient(
                    gradient,
                    self.quantization_errors[column_idx])
        end
    )

    -- Copy the quantized gradient to home
    local unquantized_gradient
    withDevice(
        home_device,
        function()
            local home_quantized_gradient =
                torch.Tensor():typeAs(remote_quantized_gradient):resizeAs(remote_quantized_gradient)
            local home_unquantizer =
                torch.Tensor():typeAs(remote_unquantizer):resizeAs(remote_unquantizer)
            self.sender_function(home_unquantizer, remote_unquantizer)
            self.sender_function(
                home_quantized_gradient, remote_quantized_gradient)
            unquantized_gradient =
                unquantize_gradient(home_quantized_gradient, home_unquantizer)
        end
    )
    return unquantized_gradient
end

function OneBitAggregator:_accumulate_quantized_gradients(gradients)
    -- We do our accumulation work on the home_device.
    local home_device = gradients[1]:getDevice()

    local unquantized_gradients = pl.tablex.pairmap(
        function(column_idx, gradient)
            return self:_transfer_gradient_to_home(
                home_device, column_idx, gradient)
        end,
        gradients
    )
    return self:_average_gradients(unquantized_gradients)
end

function OneBitAggregator:_average_gradients(gradients)
    assert(#gradients >= 1)
    local gradient_sum =
        torch.Tensor():typeAs(gradients[1]):resizeAs(gradients[1]):zero()
    pl.tablex.foreach(
        gradients,
        function(gradient) gradient_sum:add(gradient) end
    )
    -- Average the gradients across GPUs
    return gradient_sum:div(#gradients)
end

return M

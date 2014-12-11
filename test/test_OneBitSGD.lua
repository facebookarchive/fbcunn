require('fb.luaunit')
require('fbtorch')
require('cunn')

local TU = require('test.test_util')
local pl = require('pl.import_into')()
local _fbd = require('fb.debugger')
local OBSGD = require('fbcunn.OneBitSGD')

torch.setdefaulttensortype('torch.CudaTensor')

function testQuantizerOnSimpleExample()
    local gradient = torch.Tensor({{1}, {-1}})
    local quantized, unquantizer, quantization_error =
        OBSGD.quantize_gradient(
            gradient, torch.Tensor():resizeAs(gradient):zero())
    TU.assertTensorEquals(quantization_error, quantization_error:clone():zero())
    TU.assertTensorEquals(unquantizer, torch.Tensor({-1, 1}))
    TU.assertTensorEquals(quantized, torch.Tensor({{1},{0}}))
end

function testUnquantizerOnSimpleExample()
    local quantized = torch.Tensor({{1}, {0}})
    local unquantizer = torch.Tensor({{-50, 100}})
    local unquantized = OBSGD.unquantize_gradient(quantized, unquantizer)
    TU.assertTensorEquals(torch.Tensor({{100}, {-50}}), unquantized)
end

function testQuantizationReducesNormOfMatrix()
    for _ = 1,50 do
        local gradient = torch.Tensor(100, 20):normal()
        local _, _, quantization_error =
            OBSGD.quantize_gradient(
                gradient, torch.Tensor():resizeAs(gradient):zero())
        assertTrue(gradient:norm() > quantization_error:norm())
    end
end

function testAdagradWorks()
    local gradient = torch.Tensor({1})
    local adagrad_history = torch.Tensor({50})
    local base_learning_rate = 5.0
    local new_gradient, new_history =
        OBSGD.adagrad(gradient, base_learning_rate, adagrad_history)
    TU.assertTensorEquals(new_history, torch.Tensor({51}))
    TU.assertTensorAlmostEquals(new_gradient, torch.Tensor({0.70014004201}))
end


local function build_agg()
    return OBSGD.OneBitAggregator(
        {momentum_rate=1.0, adagrad_learning_rate=1.0},
        function() return torch.Tensor(5, 5):zero() end,
        function(dst, src) dst:copy(src) end
    )
end


function testOBSGDSmoothing()
    local agg = build_agg()

    local smoothed =
        agg:_smooth_gradient(agg.gradient_tensor_factory():fill(1))
    TU.assertTensorEquals(smoothed, agg.gradient_tensor_factory():fill(1))
end

function testOBSGDAveraging()
    local agg = build_agg()

    local num_columns = 5
    local gradients = pl.List.range(num_columns):map(
        function(i) return agg.gradient_tensor_factory():fill(i) end)
    local averaged_gradients = agg:_average_gradients(gradients)
    TU.assertTensorEquals(
        averaged_gradients,
        agg.gradient_tensor_factory():fill((num_columns+1) / 2)
    )
end

function testOBSGDAggregation()
    local agg = build_agg()

    local num_columns = 5
    local gradients = pl.List.range(num_columns):map(
        function(i) return agg.gradient_tensor_factory():fill(i) end)

    local averaged_gradients = agg:_accumulate_quantized_gradients(gradients)
    TU.assertTensorEquals(
        averaged_gradients,
        agg.gradient_tensor_factory():fill((num_columns+1) / 2)
    )
end


function testOBSGDEndTOEnd()
    local agg = build_agg()

    local num_columns = 5
    local gradients = pl.List.range(num_columns):map(
        function(i) return agg.gradient_tensor_factory():fill(i) end)

    local gradients_to_run = gradients:map(function(t) return t:clone() end)

    agg:run(gradients_to_run)

    TU.assertTensorEquals(
        agg.broadcast_quantization_error,
        agg.gradient_tensor_factory():zero()
    )
    TU.assertTensorEquals(
        agg.adagrad_history,
        agg.gradient_tensor_factory():fill(9)
    )
    TU.assertTensorEquals(
        agg.momentum_history,
        agg.gradient_tensor_factory():fill(1)
    )

    -- Gradients get quantized to zero
    local after_expected = agg.gradient_tensor_factory():fill(1)
    for _, t in ipairs(pl.tablex.zip(gradients, gradients_to_run)) do
        local _before, after = table.unpack(t)
        TU.assertTensorEquals(after, after_expected)
    end
end


function testMomentumWorks()
    local gradient = torch.Tensor({1})
    local momentum_history = torch.Tensor({50})
    local momentum_rate = 5.0
    local new_gradient =
        OBSGD.momentum(gradient, momentum_rate, momentum_history)
    TU.assertTensorAlmostEquals(new_gradient, torch.Tensor({251}))
end

LuaUnit:main()

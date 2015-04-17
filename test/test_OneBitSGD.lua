require('fb.luaunit')
require('fbtorch')
require('cutorch')
require('cunn')
require('fbcunn')

local TU = require('test.test_Util')
local pl = require('pl.import_into')()
local _fbd = require('fb.debugger')
local OBSGD = require('fbcunn.OneBitSGD')

torch.setdefaulttensortype('torch.CudaTensor')

 function testQuantizerOnSimpleExample()
     local gradient = torch.Tensor({{1}, {-1}})
     local accumulated = torch.Tensor():typeAs(gradient):resizeAs(gradient)
     local quantizer = nn.OneBitQuantization()
     local quantized, avg_pos, avg_neg =
         OBSGD.quantize_gradient(
             gradient, quantizer, accumulated)
     TU.assertTensorEquals(quantizer.quantization_error, quantizer.quantization_error:clone():zero())
     TU.assertTensorEquals(avg_pos, torch.Tensor({{1}, {0}}))
     TU.assertTensorEquals(avg_neg, torch.Tensor({{0}, {-1}}))
 end

 function testQuantizationReducesNormOfMatrix()
     for _ = 1,50 do
         local gradient = torch.Tensor(100, 20):normal()
         local accumulated = torch.Tensor():typeAs(gradient):resizeAs(gradient)
         local quantizer = nn.OneBitQuantization()

         OBSGD.quantize_gradient(
             gradient, quantizer, accumulated)
         assertTrue(gradient:norm() > quantizer.quantization_error:norm())
     end
 end

local function build_agg()
    return OBSGD.OneBitAggregator(
        {momentum_rate=1.0, adagrad_learning_rate=1.0},
        function() return torch.Tensor(5, 5):zero() end,
        function(dst, src) dst:copy(src) end,
        1
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
     local averaged_gradients = agg:_accumulate_quantized_gradients(gradients)
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
         agg.home_quantizer.quantization_error,
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

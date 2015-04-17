-- Copyright 2004-present Facebook. All Rights Reserved.

require('fb.luaunit')
require('fbtorch')
require('nn')
require('cutorch')
require('fbcunn')
require('fbnn')
require('ccn2')

local jac = nn.Jacobian

local function assertTensorEq(a, b, msg, precision)
   precision = precision or 1e-5
   local diff = torch.dist(a, b)
   if diff > precision then
      error('diff = ' .. diff .. ': ' .. msg)
   end
end

TestParameterValidation = {}

function TestParameterValidation:tearDown()
   collectgarbage()
end

function TestParameterValidation:testConstructorInputValidation()
   local status
   status = pcall(function() nn.LocallyConnected(1, 5, 7, 16, 6, 7) end)
   assertEquals(status, false, "Excessive kernel width not detected.")

   status = pcall(function() nn.LocallyConnected(1, 5, 7, 16, 5, 8) end)
   assertEquals(status, false, "Excessive kernel height not detected.")

   status = pcall(function() nn.LocallyConnected(0, 5, 7, 16, 5, 8) end)
   assertEquals(status, false, "Enforce > 0 input planes not detected.")

   status = pcall(function() nn.LocallyConnected(1, 0, 7, 16, 5, 8) end)
   assertEquals(status, false, "Enforce > 0 input width not detected.")

   status = pcall(function() nn.LocallyConnected(1, 5, 0, 16, 5, 8) end)
   assertEquals(status, false, "Enforce > 0 input height not detected.")

   status = pcall(function() nn.LocallyConnected(1, 5, 7, 0, 5, 8) end)
   assertEquals(status, false, "Enforce > 0 output planes not detected.")

   status = pcall(function() nn.LocallyConnected(1, 5, 7, 16, 0, 8) end)
   assertEquals(status, false, "Enforce > 0 output width not detected.")

   status = pcall(function() nn.LocallyConnected(1, 5, 7, 15, 5, 0) end)
   assertEquals(status, false, "Enforce > 0 output height not detected.")

   status = pcall(function() nn.LocallyConnected(1, 5, 7, 16, 5, 7, 0, 1) end)
   assertEquals(status, false, "Enforce column stride > 0 not detected.")

   status = pcall(function() nn.LocallyConnected(1, 5, 7, 16, 5, 7, 1, 0) end)
   assertEquals(status, false, "Enforce row stride > 0 not detected.")
end

function TestParameterValidation:testOutputSize()
   local m = nn.LocallyConnected(1, 3, 3, 1, 3, 3)
   local w, h = m:outputSize()
   assertEquals(w, 1)
   assertEquals(h, 1)

   m = nn.LocallyConnected(1, 5, 4, 1, 3, 3)
   w, h = m:outputSize()
   assertEquals(w, 3)
   assertEquals(h, 2)

   -- with stride
   m = nn.LocallyConnected(1, 3, 3, 1, 3, 3, 2, 2)
   w, h = m:outputSize()
   assertEquals(w, 1)
   assertEquals(h, 1)

   m = nn.LocallyConnected(1, 5, 8, 1, 3, 2, 2, 3)
   w, h = m:outputSize()
   assertEquals(w, 2)
   assertEquals(h, 3)
end


function TestParameterValidation:testUpdateOutputDimensions()
   local layer = nn.LocallyConnected(1, 3, 3, 1, 3, 3)

   local input = torch.Tensor(1, 3) -- 2D Tensor: bad
   local status = pcall(function() layer:updateOutput(input) end)
   assertEquals(status, false, "Invalid input size not detected.")

   input = torch.Tensor(1, 3, 3, 4, 5) -- 5D Tensor: bad, too
   status = pcall(function() layer:updateOutput(input) end)
   assertEquals(status, false, "Invalid input size not detected.")

   input = torch.Tensor(1, 3, 3) -- 3D Tensor, proper size: good
   status = pcall(function() layer:updateOutput(input) end)
   assertEquals(status, true, "Valid input caused error.")

   input = torch.Tensor(10, 1, 3, 3) -- 4D Tensor, proper size: good, too
   status = pcall(function() layer:updateOutput(input) end)
   assertEquals(status, true, "Valid input caused error.")
end

function TestParameterValidation:testUpdateOutputInputSize()
   local layer = nn.LocallyConnected(1, 2, 3, 1, 2, 3)

   local input = torch.Tensor(1, 3, 3)
   local status = pcall(function() layer:updateOutput(input) end)
   assertEquals(status, false, "Invalid input width not detected.")

   input = torch.Tensor(1, 2, 2)
   status = pcall(function() layer:updateOutput(input) end)
   assertEquals(status, false, "Invalid input height not detected.")

   input = torch.Tensor(10, 1, 3, 3)
   status = pcall(function() layer:updateOutput(input) end)
   assertEquals(status, false, "Invalid input width not detected.")

   input = torch.Tensor(10, 1, 2, 2)
   status = pcall(function() layer:updateOutput(input) end)
   assertEquals(status, false, "Invalid input height not detected.")
end



-- Initialize the tensor with a test pattern.
-- The pattern initializes the tensors linear storage
--    t[i] = start + (i-1) * delta
-- where i is the linear addess of the tensor elements.
--
local function linearPattern(tensor, start, delta)
   start = start or 1
   delta = delta or 1
   local ts = tensor:storage()
   for i = 1, ts:size() do
      ts[i] = start + (i - 1) * delta
   end
end

TestHandGilded = {}

function TestHandGilded:tearDown()
   collectgarbage()
end

-- run forward on a 1x3x3 input with 1x1x1x1x3x3 weights
function TestHandGilded:testUpdateOutput_Input1x3x3_Kernel1x3x3()
   -- test parameters
   local pI = 1
   local hI = 3
   local wI = 3

   -- create input tensor and module
   local input = torch.Tensor(pI, hI, wI)
   local layer = nn.LocallyConnected(pI, wI, hI, pI, wI, hI)

   -- initialize input, module weights, and expected value
   linearPattern(input)
   linearPattern(layer.weight)
   local e = 0
   for i = 1,9 do
      e = e + i * i
   end
   layer.bias[1][1][1] = 0.5
   local expected = torch.Tensor(1, 1, 1)
   expected[1][1][1] = e + layer.bias[1][1][1];

   local output = layer:updateOutput(input)

   -- check output value
   assertTensorEq(output, expected, "Wrong output value.", 1e-7)
end

-- run forward on a 1x3x3 input with a 1x2x2x1x2x2 weights
function TestHandGilded:testUpdateOutput_Input1x3x3_Kernel1x2x2()
   -- test parameters
   local pI = 1
   local hI = 3
   local wI = 3

   -- create input tensor and module
   local input = torch.Tensor(pI, hI, wI)
   local layer = nn.LocallyConnected(pI, wI, hI, pI, 2, 2)

   -- initialize input, module weights, and biases
   linearPattern(input)
   linearPattern(layer.weight)
   layer.bias[1] = 0.5
   local expected = torch.Tensor(1, 2, 2)

   --     |1 2 3|
   -- I = |4 5 6|
   --     |7 8 9|
   -- W =
   --     |1 2|  |5 6|  | 9 10|  |13 14|
   --     |3 4|, |7 8|, |11 12|, |15 16|
   -- b = [0.5]
   expected[1][1][1] = 0.5 +  1*1 +  2*2 + 3*4  +  4*5
   expected[1][1][2] = 0.5 +  2*5 +  3*6 + 5*7  +  6*8
   expected[1][2][1] = 0.5 + 9*4  + 10*5 + 11*7 + 12*8
   expected[1][2][2] = 0.5 + 13*5 + 14*6 + 15*8 + 16*9

   local output = layer:updateOutput(input)
   -- check output value
   assertTensorEq(output, expected, "Wrong output tensor.", 1e-7)
end

function TestHandGilded:testUpdateOutput_Input2x2x2_Kernel2x2x2()
   -- creat input tensor and module
   local input = torch.Tensor(2, 2, 2)
   local layer = nn.LocallyConnected(2, 2, 2, 1, 2, 2)

   -- initialize input, module weights, and biases
   linearPattern(input)
   linearPattern(layer.weight)
   layer.bias[1] = 0.5

   local expected = torch.Tensor(1, 1, 1)
   --             |5 6|
   -- I, W = |1 2||7 8|
   --        |3 4|
   -- b = [0.5]
   expected[1][1][1] = 0.5 + 1*1 + 2*2 + 3*3 + 4*4 + 5*5 + 6*6 + 7*7 + 8*8

   local output = layer:updateOutput(input)
   -- check result
   assertTensorEq(output, expected, "Wrong output tensor.", 1e-7)
end

function TestHandGilded:testUpdateOutput_Input2x2x2_Kernel2x2x2()
   -- creat input tensor and module
   local input = torch.Tensor(2, 2, 2)
   local layer = nn.LocallyConnected(2, 2, 2, 2, 2, 2)

   -- initialize input, module weights, and biases
   linearPattern(input)
   linearPattern(layer.weight)
   linearPattern(layer.bias, 0.5)

   local expected = torch.Tensor(2, 1, 1)
   --          |5 6|          |5 6|         |13 14|
   -- I = |1 2||7 8| W = |1 2||7 8|  | 9 10||15 16|
   --     |3 4|          |3 4|     , |11 12|
   -- b = [0.5, 1.5]
   expected[1][1][1] = 0.5 + 1*1  + 2*2  + 3*3  + 4*4
      + 5*5  + 6*6  + 7*7  + 8*8
   expected[2][1][1] = 1.5 + 1*9  + 2*10 + 3*11 + 4*12
      + 5*13 + 6*14 + 7*15 + 8*16

   local output = layer:updateOutput(input)
   -- check result
   assertTensorEq(output, expected, "Wrong output tensor.", 1e-7)
end

-- run forward on a 1x4x4 input with a 1x2x2x1x2x2 weights 2x2 stride
function TestHandGilded:testUpdateOutput_Input1x4x4_Kernel1x2x2_Stride2x2()
   -- test parameters
   local pI = 1
   local hI = 4
   local wI = 4

   -- create input tensor and module
   local input = torch.Tensor(pI, hI, wI)
   local layer = nn.LocallyConnected(pI, wI, hI, pI, 2, 2, 2, 2)

   -- initialize input, module weights, and biases
   linearPattern(input)
   linearPattern(layer.weight)
   layer.bias[1] = 0.5
   local expected = torch.Tensor(1, 2, 2)

   --     | 1  2  3  4|
   -- I = | 5  6  7  8|
   --     | 9 10 11 12|
   --     |13 14 15 16|
   -- W =
   --     |1 2|  |5 6|  | 9 10|  |13 14|
   --     |3 4|, |7 8|, |11 12|, |15 16|
   -- b = [0.5]
   expected[1][1][1] = 0.5 +  1*1  +  2*2  +  3*5  +  4*6
   expected[1][1][2] = 0.5 +  5*3  +  6*4  +  7*7  +  8*8
   expected[1][2][1] = 0.5 +  9*9  + 10*10 + 11*13 + 12*14
   expected[1][2][2] = 0.5 + 13*11 + 14*12 + 15*15 + 16*16

   local output = layer:updateOutput(input)
   -- check output value
   assertTensorEq(output, expected, "Wrong output tensor.", 1e-7)
end

local function setUpTest(iP, iH, iW, oP, kH, kW, dH, dW)
   local input = torch.Tensor(iP, iH, iW):uniform(0, 1)
   -- Note: The constructor takes tensor parameter in order width-height
   local layer = nn.LocallyConnected(iP, iW, iH, oP, kW, kH, dW, dH)

   return input, layer
end

local function setUpBatchTest(batch_size, iP, iH, iW, oP, kH, kW, dH, dW)
   local input = torch.Tensor(batch_size, iP, iH, iW):uniform(0, 1)
   local layer = nn.LocallyConnected(iP, iW, iH, oP, kW, kH, dW, dH)

   return input, layer
end


TestJacobian = {}

function TestJacobian:tearDown()
   collectgarbage()
end

function TestJacobian:testUpdateGradInputJacobian()
   local input, layer = setUpTest(4, 8, 8, 2, 5, 5)
   local err = jac.testJacobian(layer, input)
   assertAlmostEquals(err, 0.0, 1e-7)
end

function TestJacobian:testUpdateGradInputJacobianBatch()
   local input, layer = setUpBatchTest(2, 4, 8, 8, 2, 5, 5)
   local err = jac.testJacobian(layer, input)
   assertAlmostEquals(err, 0.0, 1e-7)
end

function TestJacobian:testUpdateGradInputJacobianStrided()
   local input, layer = setUpTest(4, 13, 13, 2, 5, 5, 2, 2)
   local err = jac.testJacobian(layer, input)
   assertAlmostEquals(err, 0.0, 1e-7)
end

function TestJacobian:testAccGradParametersJacobianWeights()
   local input, layer = setUpTest(4, 8, 8, 2, 5, 5)
   local err = jac.testJacobianParameters(layer, input, layer.weight,
                                          layer.gradWeight)
   assertAlmostEquals(err, 0.0, 1e-7)
end

function TestJacobian:testAccGradParametersJacobianWeightsBatch()
   local input, layer = setUpBatchTest(2, 4, 8, 8, 2, 5, 5)
   local err = jac.testJacobianParameters(layer, input, layer.weight,
                                          layer.gradWeight)
   assertAlmostEquals(err, 0.0, 1e-7)
end

function TestJacobian:testAccGradParametersJacobianWeightsStrided()
   local input, layer = setUpTest(4, 13, 13, 2, 5, 5, 2, 2)
   local err = jac.testJacobianParameters(layer, input, layer.weight,
                                          layer.gradWeight)
   assertAlmostEquals(err, 0.0, 1e-7)
end

function TestJacobian:testAccGradParametersJacobianBiases()
   local input, layer = setUpTest(4, 8, 8, 2, 5, 5)
   local err = jac.testJacobianParameters(layer, input, layer.bias,
                                          layer.gradBias)
   assertAlmostEquals(err, 0.0, 1e-7)
end

function TestJacobian:testAccGradParametersJacobianBiasesBatch()
   local input, layer = setUpBatchTest(2, 4, 8, 8, 2, 5, 5)
   local err = jac.testJacobianParameters(layer, input, layer.bias,
                                          layer.gradBias)
   assertAlmostEquals(err, 0.0, 1e-7)
end

function TestJacobian:testAccGradParametersJacobianBiasesStrided()
   local input, layer = setUpTest(4, 13, 13, 2, 5, 5, 2, 2)
   local err = jac.testJacobianParameters(layer, input, layer.bias,
                                          layer.gradBias)
   assertAlmostEquals(err, 0.0, 1e-7)
end

function TestJacobian:testAccGradParametersJacobianUpdateWeights()
   local input, layer = setUpTest(4, 8, 8, 2, 5, 5)
   local err = jac.testJacobianUpdateParameters(layer, input, layer.weight)
   assertAlmostEquals(err, 0.0, 1e-7)
end

function TestJacobian:testAccGradParametersJacobianUpdateWeightsBatch()
   local input, layer = setUpBatchTest(2, 4, 8, 8, 2, 5, 5)
   local err = jac.testJacobianUpdateParameters(layer, input, layer.weight)
   assertAlmostEquals(err, 0.0, 1e-7)
end

function TestJacobian:testAccGradParametersJacobianUpdateWeightsStrided()
   local input, layer = setUpTest(4, 13, 13, 2, 5, 5, 2, 2)
   local err = jac.testJacobianUpdateParameters(layer, input, layer.weight)
   assertAlmostEquals(err, 0.0, 1e-7)
end

function TestJacobian:testAccGradParametersJacobianUpdateBiases()
   local input, layer = setUpTest(4, 8, 8, 2, 5, 5)
   local output = layer:forward(input)
   layer:zeroGradParameters()
   layer:backward(input, output:clone():fill(1))

   local err = jac.testJacobianUpdateParameters(layer, input, layer.bias)
   assertAlmostEquals(err, 0.0, 1e-7)
end

function TestJacobian:testAccGradParametersJacobianUpdateBiasesBatch()
   local input, layer = setUpBatchTest(2, 4, 8, 8, 2, 5, 5)
   local output = layer:forward(input)
   layer:zeroGradParameters()
   layer:backward(input, output:clone():fill(1))

   local err = jac.testJacobianUpdateParameters(layer, input, layer.bias)
   assertAlmostEquals(err, 0.0, 1e-7)
end

function TestJacobian:testAccGradParametersJacobianUpdateBiasesStrided()
   local input, layer = setUpTest(4, 13, 13, 2, 5, 5, 2, 2)
   local output = layer:forward(input)
   layer:zeroGradParameters()
   layer:backward(input, output:clone():fill(1))

   local err = jac.testJacobianUpdateParameters(layer, input, layer.bias)
   assertAlmostEquals(err, 0.0, 1e-7)
end

local function CudaConstructor(P_i, H_i, W_i, P_o, H_k, W_k)
   -- make LocallyConnected float layer
   local layer = nn.LocallyConnected(P_i, W_i, H_i, P_o, W_k, H_k):float()
   local W_o, H_o = layer:outputSize()
   assertEquals(P_o, layer.weight:size(1)) -- output planes
   assertEquals(H_o, layer.weight:size(2)) -- output height
   assertEquals(W_o, layer.weight:size(3)) -- output width
   assertEquals(P_i, layer.weight:size(4)) -- input planes
   assertEquals(H_k, layer.weight:size(5)) -- kernel height
   assertEquals(W_k, layer.weight:size(6)) -- kernel width
   assert(layer.weight:isContiguous(),
          'Layer.weight must always be contiguous.')

   local defaultTorchType = torch.getdefaulttensortype()
   -- if the LuaTest framework runs multi-threaded by default, then
   -- this might be a problem, since the default tensor type is a global
   -- constant.
   torch.setdefaulttensortype('torch.CudaTensor')
   local layerCUDA = nn.LocallyConnected(P_i, W_i, H_i, P_o, W_k, H_k)
   W_o, H_o = layerCUDA:outputSize()
   assertEquals(H_o, layerCUDA.weight:size(1))
   assertEquals(W_o, layerCUDA.weight:size(2))
   assertEquals(H_k, layerCUDA.weight:size(3))
   assertEquals(W_k, layerCUDA.weight:size(4))
   assertEquals(P_o, layerCUDA.weight:size(5))
   assertEquals(P_i, layerCUDA.weight:size(6))
   assert(layerCUDA.weight:isContiguous(),
          'Layer.weight must always be contiguous.')
   -- restore default tensor type
   torch.setdefaulttensortype(defaultTorchType)
end

TestTensorLayoutConversion = {}

function TestTensorLayoutConversion:tearDown()
   collectgarbage()
end

local function createReversibilityTest(suite, tensorType)
   local defaulttensortype = torch.getdefaulttensortype()
   torch.setdefaulttensortype(tensorType)

   suite['testReversability3D' .. tensorType] = function(s)
      local tensor = torch.Tensor(3, 5, 7):uniform(0, 1)
      -- assert that toInterleaved reverses toPlanar
      assertTensorEq(tensor, nn.LocallyConnected.toPlanar(
                        nn.LocallyConnected.toInterleaved(tensor, false),
                        false))
      assertTensorEq(tensor, nn.LocallyConnected.toPlanar(
                        nn.LocallyConnected.toInterleaved(tensor, true), false))
      assertTensorEq(tensor, nn.LocallyConnected.toPlanar(
                        nn.LocallyConnected.toInterleaved(tensor, false), true))
      assertTensorEq(tensor, nn.LocallyConnected.toPlanar(
                        nn.LocallyConnected.toInterleaved(tensor, true), true))
      -- assert that toPlanar reverses toInterleave
      assertTensorEq(tensor, nn.LocallyConnected.toInterleaved(
                        nn.LocallyConnected.toPlanar(tensor, false), false))
      assertTensorEq(tensor, nn.LocallyConnected.toInterleaved(
                        nn.LocallyConnected.toPlanar(tensor, true), false))
      assertTensorEq(tensor, nn.LocallyConnected.toInterleaved(
                        nn.LocallyConnected.toPlanar(tensor, false), true))
      assertTensorEq(tensor, nn.LocallyConnected.toInterleaved(
                        nn.LocallyConnected.toPlanar(tensor, true), true))
   end

   suite['testReversability4D' .. tensorType] = function(s)
      local tensor = torch.Tensor(3, 5, 7, 11):uniform(0, 1)
      -- assert that toInterleaved reverses toPlanar
      assertTensorEq(tensor, nn.LocallyConnected.toPlanar(
                        nn.LocallyConnected.toInterleaved(tensor, false),
                        false))
      assertTensorEq(tensor, nn.LocallyConnected.toPlanar(
                        nn.LocallyConnected.toInterleaved(tensor, true), false))
      assertTensorEq(tensor, nn.LocallyConnected.toPlanar(
                        nn.LocallyConnected.toInterleaved(tensor, false), true))
      assertTensorEq(tensor, nn.LocallyConnected.toPlanar(
                        nn.LocallyConnected.toInterleaved(tensor, true), true))
      -- assert that toPlanar reverses toInterleave
      assertTensorEq(tensor, nn.LocallyConnected.toInterleaved(
                        nn.LocallyConnected.toPlanar(tensor, false), false))
      assertTensorEq(tensor, nn.LocallyConnected.toInterleaved(
                        nn.LocallyConnected.toPlanar(tensor, true), false))
      assertTensorEq(tensor, nn.LocallyConnected.toInterleaved(
                        nn.LocallyConnected.toPlanar(tensor, false), true))
      assertTensorEq(tensor, nn.LocallyConnected.toInterleaved(
                        nn.LocallyConnected.toPlanar(tensor, true), true))
   end

   suite['testReversability6D' .. tensorType] = function(s)
      local tensor = torch.Tensor(3, 5, 7, 11, 13, 17):uniform(0, 1)
      -- assert that toInterleaved reverses toPlanar
      assertTensorEq(tensor, nn.LocallyConnected.toPlanar(
                        nn.LocallyConnected.toInterleaved(tensor, false),
                        false))
      assertTensorEq(tensor, nn.LocallyConnected.toPlanar(
                        nn.LocallyConnected.toInterleaved(tensor, true), false))
      assertTensorEq(tensor, nn.LocallyConnected.toPlanar(
                        nn.LocallyConnected.toInterleaved(tensor, false), true))
      assertTensorEq(tensor, nn.LocallyConnected.toPlanar(
                        nn.LocallyConnected.toInterleaved(tensor, true), true))
      -- assert that toPlanar reverses toInterleave
      assertTensorEq(tensor, nn.LocallyConnected.toInterleaved(
                        nn.LocallyConnected.toPlanar(tensor, false), false))
      assertTensorEq(tensor, nn.LocallyConnected.toInterleaved(
                        nn.LocallyConnected.toPlanar(tensor, true), false))
      assertTensorEq(tensor, nn.LocallyConnected.toInterleaved(
                        nn.LocallyConnected.toPlanar(tensor, false), true))
      assertTensorEq(tensor, nn.LocallyConnected.toInterleaved(
                        nn.LocallyConnected.toPlanar(tensor, true), true))
   end

   torch.setdefaulttensortype(defaulttensortype)
end

-- instantiate reversibility test suites
createReversibilityTest(TestTensorLayoutConversion,
                        torch.getdefaulttensortype())
createReversibilityTest(TestTensorLayoutConversion, 'torch.CudaTensor')

function TestTensorLayoutConversion:testConstructor6x11x13x4x5x3()
   CudaConstructor(6, 11, 13, 4, 5, 3)
end

function TestTensorLayoutConversion:testFloatToCudaConversion()
   -- make LocallyConnected float layer
   local P_i = 6  -- output planes
   local H_i = 11 -- output height
   local W_i = 13 -- output width
   local P_o = 4  -- input planes
   local H_k = 5  -- kernel height
   local W_k = 3  -- kernel width
   local layer = nn.LocallyConnected(P_i, W_i, H_i, P_o, W_k, H_k):float()
   local W_o, H_o = layer:outputSize()
   assertEquals(P_o, layer.weight:size(1))
   assertEquals(H_o, layer.weight:size(2))
   assertEquals(W_o, layer.weight:size(3))
   assertEquals(P_i, layer.weight:size(4))
   assertEquals(H_k, layer.weight:size(5))
   assertEquals(W_k, layer.weight:size(6))
   assert(layer.weight:isContiguous(),
          'Layer.weight must always be contigous.')
   -- convert layer to CUDA
   layer:cuda()
   assertEquals(H_o, layer.weight:size(1))
   assertEquals(W_o, layer.weight:size(2))
   assertEquals(H_k, layer.weight:size(3))
   assertEquals(W_k, layer.weight:size(4))
   assertEquals(P_o, layer.weight:size(5))
   assertEquals(P_i, layer.weight:size(6))
   assert(layer.weight:isContiguous(),
          'Layer.weight must always be contigous.')
end

function TestTensorLayoutConversion:testCudaToFloatConversion()
   local P_i = 6  -- output planes
   local H_i = 11 -- output height
   local W_i = 13 -- output width
   local P_o = 4  -- input planes
   local H_k = 5  -- kernel height
   local W_k = 3  -- kernel width
   local layer = nn.LocallyConnected(P_i, W_i, H_i, P_o, W_k, H_k):cuda()
   local W_o, H_o = layer:outputSize()
   assertEquals(H_o, layer.weight:size(1))
   assertEquals(W_o, layer.weight:size(2))
   assertEquals(H_k, layer.weight:size(3))
   assertEquals(W_k, layer.weight:size(4))
   assertEquals(P_o, layer.weight:size(5))
   assertEquals(P_i, layer.weight:size(6))
   assert(layer.weight:isContiguous(),
          'Layer.weight must always be contigous.')
   -- convert layer to CUDA
   layer:float()
   assertEquals(P_o, layer.weight:size(1))
   assertEquals(H_o, layer.weight:size(2))
   assertEquals(W_o, layer.weight:size(3))
   assertEquals(P_i, layer.weight:size(4))
   assertEquals(H_k, layer.weight:size(5))
   assertEquals(W_k, layer.weight:size(6))
   assert(layer.weight:isContiguous(),
          'Layer.weight must always be contigous.')
end

function TestTensorLayoutConversion:testWeightConversion()
   -- layer with random weights
   local layer1 = nn.LocallyConnected(8, 11, 13, 4, 3, 5):float()
   layer1:reset()
   -- make a clone and convert to cuda and back
   local layer2 = layer1:clone():cuda():float()
   assertTensorEq(layer1.weight, layer2.weight, 'weight tensors not identical')
end

local function updateOutputCUDA(pi, hi, wi, po, hk, wk, hd, wd, eps)
   hd = hd or 1
   wd = wd or 1
   eps = eps or 1e-5
   local input, layer = setUpTest(pi, hi, wi, po, hk, wk, hd, wd)
   input = input:float()
   assert(input:isContiguous(), 'New tensor must be contiguous')
   assert(input:size(1) == pi and input:size(2) == hi and input:size(3) == wi,
          'Wrong input size.')
   layer:float()
   local output = layer:forward(input)
   local inputCUDA = input:type('torch.CudaTensor')
   assert(inputCUDA:isContiguous(), 'New tensor must be contiguous')
   local layerCUDA = layer:clone():cuda()
   local outputCUDA = layerCUDA:forward(inputCUDA)
   assert(inputCUDA:isContiguous(), 'forward() must not change input')
   for dim = 1,3 do
      assert(output:size(dim) == outputCUDA:size(dim), 'Output size mismatch.')
      assert(input:size(dim) == inputCUDA:size(dim), 'Input size mismatch.')
   end
   -- Check that sufficiently complex tensors come out non-contiguous unless
   -- outputs are forced to be.
   if layerCUDA.forceContiguous then
      assert(outputCUDA:isContiguous())
   else
      -- The comparison to the transposed host tensor is necessary,
      -- because not all transpositions are non-contiguous.
      assert(outputCUDA:isContiguous() ==
                output:transpose(2, 3):transpose(1, 3):isContiguous(),
             'forward()\'s output must not be contigous')
   end
   local outputHost = outputCUDA:float()
   assertTensorEq(output, outputHost,
                  'LocallyConnected CUDA doesn\'t match reference.', eps)
end

local function updateOutputBatchCUDA(bi, pi, hi, wi, po, hk, wk, hd, wd, eps)
   hd = hd or 1
   wd = wd or 1
   eps = eps or 1e-5
   local input, layer = setUpBatchTest(bi, pi, hi, wi, po, hk, wk, hd, wd)
   input = input:float()
   layer:float()
   local output = layer:forward(input)
   local inputCUDA = input:type('torch.CudaTensor')
   local layerCUDA = layer:clone():cuda()
   local outputCUDA = layerCUDA:forward(inputCUDA)
   assert(inputCUDA:isContiguous(), 'forward() must not change input')
   for dim = 1,4 do
      assert(output:size(dim) == outputCUDA:size(dim), 'Output size mismatch.')
      assert(input:size(dim) == inputCUDA:size(dim), 'Input size mismatch.')
   end
   -- Check that sufficiently complex tensors come out non-contiguous unless
   -- outputs are forced to be.
   if layerCUDA.forceContiguous then
      assert(outputCUDA:isContiguous())
   else
      -- The comparison to the transposed host tensor is necessary,
      -- because not all transpositions are non-contiguous.
      assert(outputCUDA:isContiguous() ==
                output:transpose(3, 4):transpose(2, 4):isContiguous(),
             'forward()\'s output must not be contigous')
   end
   local outputHost = outputCUDA:float()
   assertTensorEq(output, outputHost,
                  'LocallyConnected CUDA doesn\'t match reference.', eps)
end

TestUpdateOutputCUDA = {}

function TestUpdateOutputCUDA:tearDown()
   collectgarbage()
end

-- test on 1x1x1 input image, 1x1x1 kernel, producing a 32-channel single
-- pixel output image
function TestUpdateOutputCUDA:testInput1x1x1_Kernel32x1x1()
   updateOutputCUDA(1, 1, 1, 32, 1, 1)
end
function TestUpdateOutputCUDA:testBatch5_Input1x1x1_Kernel32x1x1()
   updateOutputBatchCUDA(5, 1, 1, 1, 32, 1, 1)
end

-- test on a 1x1x2 input image, 1x1x1 kernel, producing a 32-channel two
-- pixel output image.
function TestUpdateOutputCUDA:testInput1x1x2_Kernel32x1x1()
   updateOutputCUDA(1, 1, 2, 32, 1, 1)
end
function TestUpdateOutputCUDA:testBatch5_Input1x1x2_Kernel32x1x1()
   updateOutputBatchCUDA(5, 1, 1, 2, 32, 1, 1)
end

-- test on a 1x2x1 input image, 1x1x1 kernel, producing a 32-channel two
-- pixel output image.
function TestUpdateOutputCUDA:testInput1x2x1_Kernel32x1x1()
   updateOutputCUDA(1, 2, 1, 32, 1, 1)
end
function TestUpdateOutputCUDA:testBatch5_Input1x2x1_Kernel32x1x1()
   updateOutputBatchCUDA(5, 1, 2, 1, 32, 1, 1)
end
-- testUpdateOutputBatchCUDA1x2x1x32x1x1()

-- test on a 1x1x2 input image, 1x1x2 kernel, producing a 32-channel single
-- pixel output image.
function TestUpdateOutputCUDA:testInput1x1x2_Kernel32x1x2()
   updateOutputCUDA(1, 1, 2, 32, 1, 2)
end
function TestUpdateOutputCUDA:testBatch5_Input1x1x2_Kernel32x1x2()
   updateOutputBatchCUDA(5, 1, 1, 2, 32, 1, 2)
end

-- test on a 1x1x64 input image, 1x1x9 kernel, producing a 16-channel, 56
-- pixel output image.
function TestUpdateOutputCUDA:testInput1x1x64_Kernel16x1x9()
   updateOutputCUDA(1, 1, 64, 16, 1, 9)
end

function TestUpdateOutputCUDA:testBatch5_Input1x1x64_Kernel16x1x9()
   updateOutputBatchCUDA(5, 1, 1, 64, 16, 1, 9, 1, 1, 2e-5)
end

-- test on a 1x2x1 input image, 1x2x1 kernel, producing a 32-channel single
-- pixel output image.
function TestUpdateOutputCUDA:testInput1x2x1_Kernel32x2x1()
   updateOutputCUDA(1, 2, 1, 32, 2, 1)
end
function TestUpdateOutputCUDA:testBatch5_Input1x2x1_Kernel32x2x1()
   updateOutputBatchCUDA(5, 1, 2, 1, 32, 2, 1)
end

-- test on a 1x64x1 input image, 1x9x1 kernel, producing a 16-channel, 56x1
-- pixel output image.
function TestUpdateOutputCUDA:testInput1x64x1_Kernel16x9x1()
   updateOutputCUDA(1, 64, 1, 16, 9, 1)
end
function TestUpdateOutputCUDA:testBatch5_Input1x64x1_Kernel16x9x1()
      updateOutputBatchCUDA(5, 1, 64, 1, 16, 9, 1)
end

-- test on a 1x1x64 input image, 1x1x9 kernel, producing a 16-channel, 1x56
-- pixel output image.
function TestUpdateOutputCUDA:testInput1x1x64_Kernel16x1x9()
   updateOutputCUDA(1, 1, 64, 16, 1, 9, 1, 1, 2e-5)
end
function TestUpdateOutputCUDA:testBatch5_Input1x1x64_Kernel16x1x9()
   updateOutputBatchCUDA(5, 1, 1, 64, 16, 1, 9, 1, 1, 2e-3)
end

-- test on a 2x1x1 input image, 2x1x1 kernel, producing a 16-channel single
-- pixel output image.
function TestUpdateOutputCUDA:testInput2x1x1_Kernel16x1x1()
   updateOutputCUDA(2, 1, 1, 16, 1, 1)
end
function TestUpdateOutputCUDA:testBatch5_Input2x1x1_Kernel16x1x1()
   updateOutputBatchCUDA(5, 2, 1, 1, 16, 1, 1)
end

-- test on a 2x1x1 input image, 2x1x1 kernel, producing a 32-channel single
-- pixel output image.
function TestUpdateOutputCUDA:testInput2x1x1_Kernel32x1x1()
   updateOutputCUDA(2, 1, 1, 32, 1, 1)
end
function TestUpdateOutputCUDA:testBatch5_Input2x1x1_Kernel32x1x1()
   updateOutputBatchCUDA(5, 2, 1, 1, 32, 1, 1)
end

-- test on a 16x1x1 input image, 16x1x1 kernel, producing a 16-channel single
-- pixel output image.
function TestUpdateOutputCUDA:testInput16x1x1_Kernel16x1x1()
   updateOutputCUDA(16, 1, 1, 16, 1, 1)
end
function TestUpdateOutputCUDA:testBatch5_Input16x1x1_Kernel16x1x1()
   updateOutputBatchCUDA(5, 16, 1, 1, 16, 1, 1)
end

-- test on a 16x11x13 input image, 16x5x7 kernel, producing a 16-channel,
-- 7x8 output image.
function TestUpdateOutputCUDA:testInput16x11x13_Kernel16x5x7()
   updateOutputCUDA(16, 11, 13, 16, 5, 7)
end
function TestUpdateOutputCUDA:testBatch5_Input16x11x13_Kernel16x5x7()
   updateOutputBatchCUDA(5, 16, 11, 13, 16, 5, 7)
end

-- test on a 32x11x13 input image, 32x5x7 kernel, producing a 32-channel,
-- 7x8 output image.
function TestUpdateOutputCUDA:testInput32x11x13_Planes32_Kernel5x7()
   updateOutputCUDA(32, 11, 13, 32, 5, 7)
end
function TestUpdateOutputCUDA:testBatch5_Input32x11x13_Planes32_Kernelx1x1()
   updateOutputBatchCUDA(5, 32, 11, 13, 32, 5, 7, 1, 1, 2e-5)
end

-- test "small, square kernel' optimization with a 32x44x44 input image,
-- 32x3x3 kernel, producing a 32-channel, 42x42 output image.
function TestUpdateOutputCUDA:testInput32x44x44_Kerne32x3x3()
   updateOutputCUDA(32, 44, 44, 32, 3, 3, 1, 1, 3e-5)
end
function TestUpdateOutputCUDA:testBatch5_Input32x44x44_Kernel32x3x3()
   updateOutputBatchCUDA(5, 32, 44, 44, 32, 3, 3, 1, 1, 7e-5)
end

-- test that iterating kernel is called for inputs larger than what a
-- single warp can handle that happen to be powers of two.
function TestUpdateOutputCUDA:testInput64x11x13_Planes4_Kernel5x7()
   updateOutputCUDA(64, 11, 13, 4, 5, 7)
end
function TestUpdateOutputCUDA:testBatch5_Input64x11x13_Planes4_Kernelx1x1()
   updateOutputBatchCUDA(5, 64, 11, 13, 4, 5, 7, 1, 1, 2e-5)
end

-- officially support up to 256 planes on inputs and kernels
-- test on a 256x11x13 input image  256x5x7 kernel, producing a 256-channel,
-- 7x8 output image.
function TestUpdateOutputCUDA:testInput256x11x13_Planes256_Kernel5x7()
   updateOutputCUDA(256, 11, 13, 256, 5, 7, 1, 1, 9e-5)
end
function TestUpdateOutputCUDA:testBatch5_Input256x11x13_Planes256_Kernelx1x1()
   updateOutputBatchCUDA(5, 256, 11, 13, 256, 5, 7, 1, 1, 19e-5)
end

-- test on a 64x11x13 input image, 64x5x7 kernel, producing a 32-channel,
-- 7x8 output image.
function TestUpdateOutputCUDA:testInput64x11x13_Planes64_Kernel5x7()
   updateOutputCUDA(64, 11, 13, 64, 5, 7, 1, 1, 2e-5)
end
function TestUpdateOutputCUDA:testBatch5_Input64x11x13_Planes64_Kernelx1x1()
   updateOutputBatchCUDA(5, 64, 11, 13, 64, 5, 7, 1, 1, 4e-5)
end


-- -----------------------------------------------------------------------------
-- non-power-of-two tests
-- -----------------------------------------------------------------------------

-- test on a 5x11x13 input image, 4x5x7 kernel, producing a 4-channel,
-- 7x8 output image.
-- Test that the iterating kernel variant works with small non-power-of-two
-- input planes.
function TestUpdateOutputCUDA:testInput5x11x13_Planes4_Kernel5x7()
   updateOutputCUDA(5, 11, 13, 4, 5, 7)
end
function TestUpdateOutputCUDA:testBatch5_Input5x11x13_Planes4_Kernelx5x7()
   updateOutputBatchCUDA(5, 5, 11, 13, 4, 5, 7)
end

-- test on a 24x11x13 input image, 16x5x7 kernel, producing a 16-channel,
-- 7x8 output image.
-- Test that the iterating kernel variant work with arbitrary numbers of
-- input planes (in particularly non-powers-of-two or multiples of 32).
function TestUpdateOutputCUDA:testInput24x11x13_Planes16_Kernel5x7()
   updateOutputCUDA(24, 11, 13, 16, 5, 7)
end
function TestUpdateOutputCUDA:testBatch5_Input24x11x13_Planes16_Kernelx5x7()
   updateOutputBatchCUDA(5, 24, 11, 13, 16, 5, 7, 1, 1, 3e-5)
end

-- test on a 39x11x13 input image, 32x5x7 kernel, producing a 32-channel,
-- 7x8 output image.
-- Test that the iterating kernel variant work with arbitrary numbers of
-- input planes (in particularly non-powers-of-two or multiples of 32).
function TestUpdateOutputCUDA:testInput39x11x13_Planes32_Kernel5x7()
   updateOutputCUDA(39, 11, 13, 32, 5, 7)
end
function TestUpdateOutputCUDA:testBatch5_Input39x11x13_Planes32_Kernelx5x7()
   updateOutputBatchCUDA(5, 39, 11, 13, 32, 5, 7, 1, 1, 3e-5)
end

-- test on a 32x11x13 input image, 39x5x7 kernel, producing a 39-channel,
-- 7x8 output image.
-- Test that the iterating kernel variant work with arbitrary numbers of
-- output planes > 32.
function TestUpdateOutputCUDA:testInput32x11x13_Planes39_Kernel5x7()
   updateOutputCUDA(32, 11, 13, 39, 5, 7, 1, 1, 2e-5)
end
function TestUpdateOutputCUDA:testBatch5_Input32x11x13_Planes39_Kernelx5x7()
   updateOutputBatchCUDA(5, 32, 11, 13, 39, 5, 7, 1, 1, 3e-5)
end


Facer = {}

function Facer:tearDown()
   collectgarbage()
end

-- test on the local3 parameters from Facer
function Facer:test_local3_updateOutput()
   updateOutputCUDA(16, 64, 64, 16, 9, 9, 1, 1, 6e-5)
end
function Facer:test_local3batch_updateOutputBatch()
   updateOutputBatchCUDA(5, 16, 64, 64, 16, 9, 9, 1, 1, 6e-3)
end

-- test on the local4 parameters from Facer
function Facer:test_local4_updateOutput()
   updateOutputCUDA(16, 55, 55, 16, 7, 7, 2, 2, 2e-5)
end
function Facer:test_local4_updateOutputBatch()
   updateOutputBatchCUDA(5, 16, 55, 55, 16, 7, 7, 2, 2, 2e-4)
end

--test on the local5 parameters from Facer
function Facer:test_local5_updateOutput()
   updateOutputCUDA(16, 24, 24, 16, 5, 5, 1, 1, 2e-4)
end
function Facer:test_local5_updateOutputBatch()
   updateOutputBatchCUDA(5, 16, 24, 24, 16, 5, 5, 1, 1, 2e-4)
end

local function updateGradInputCUDA(pi, hi, wi, po, hk, wk, hd, wd, eps)
   collectgarbage()
   hd = hd or 1
   wd = wd or 1
   eps = eps or 1e-5
   local input, layer = setUpTest(pi, hi, wi, po, hk, wk, hd, wd)
   input = input:float()
   layer:float()
   local output = layer:forward(input)
   output:apply(function(x) return torch.uniform() end)
   local gradInput = layer:updateGradInput(input, output)
   local inputCUDA = input:type('torch.CudaTensor')
   local layerCUDA = layer:clone():cuda()
   layerCUDA:forward(inputCUDA)
   local outputCUDA = output:cuda()
   local gradInputCUDA = layerCUDA:updateGradInput(inputCUDA, outputCUDA)
   assert(inputCUDA:isContiguous(),
          'updateGradInput() must not change its inputs')
   assert(outputCUDA:isContiguous(),
          'updateGradInput() must not change its inputs')
   for dim = 1,3 do
      assert(gradInput:size(dim) == gradInputCUDA:size(dim),
             'Result size mismatch.')
      assert(output:size(dim) == outputCUDA:size(dim), 'Input size mismatch.')
   end
   -- Check that sufficiently complex tensors come out non-contiguous unless
   -- outputs are forced to be.
   if layerCUDA.forceContiguous then
      assert(gradInputCUDA:isContiguous())
   else
      -- The comparison to the transposed host tensor is necessary,
      -- because not all transpositions are non-contiguous.
      assert(gradInputCUDA:isContiguous() ==
                gradInput:transpose(2, 3):transpose(1, 3):isContiguous(),
             'forward()\'s output must not be contigous')
   end
   gradInputCUDA = gradInputCUDA:float()
   assertTensorEq(gradInput, gradInputCUDA,
                  'LocallyConnected CUDA doesn\'t match reference.', eps)
end

local function updateGradInputBatchCUDA(bi, pi, hi, wi, po, hk, wk, hd, wd, eps)
   collectgarbage()
   hd = hd or 1
   wd = wd or 1
   eps = eps or 1e-5

   local input, layer = setUpBatchTest(bi, pi, hi, wi, po, hk, wk, hd, wd)
   input = input:float()
   layer:float()
   local output = layer:forward(input)
   output:apply(function(x) return torch.uniform() end)
   local gradInput = layer:updateGradInput(input, output)
   local inputCUDA = input:type('torch.CudaTensor')
   assert(inputCUDA:isContiguous(),
          'Default constructed CUDA tensor must be contiguous')
   local layerCUDA = layer:clone():cuda()
   layerCUDA:forward(inputCUDA)
   local outputCUDA = output:cuda()
   assert(outputCUDA:isContiguous(),
          'Default constructed CUDA tensor must be contiguous')
   local gradInputCUDA = layerCUDA:updateGradInput(inputCUDA, outputCUDA)
   assert(inputCUDA:isContiguous(),
          'updateGradInput() must not change its inputs')
   assert(outputCUDA:isContiguous(),
          'updateGradInput() must not change its inputs')
   for dim = 1,4 do
      assert(gradInput:size(dim) == gradInputCUDA:size(dim),
             'Result size mismatch.')
      assert(output:size(dim) == outputCUDA:size(dim), 'Input size mismatch.')
   end
   if layerCUDA.forceContiguous then
      assert(gradInputCUDA:isContiguous())
   else
      -- The comparison to the transposed host tensor is necessary,
      -- because not all transpositions are non-contiguous.
      assert(gradInputCUDA:isContiguous() ==
                gradInput:transpose(3, 4):transpose(2, 4):isContiguous(),
             'forward()\'s output must not be contigous')
   end
   gradInputCUDA = gradInputCUDA:float()
   assertTensorEq(gradInput, gradInputCUDA,
                  'LocallyConnected CUDA doesn\'t match reference.', eps)
end


TestUpdateGradInputCUDA = {}

function TestUpdateGradInputCUDA:testInput1x1x1_Kernel32x1x1()
   updateGradInputCUDA(1, 1, 1, 32, 1, 1)
end
function TestUpdateGradInputCUDA:testBatch5_Input1x1x1_Kernel32x1x1()
   updateGradInputBatchCUDA(5, 1, 1, 1, 32, 1, 1)
end

function TestUpdateGradInputCUDA:testInput1x1x2_Kernel32x1x1()
   updateGradInputCUDA(1, 1, 2, 32, 1, 1)
end
function TestUpdateGradInputCUDA:testBatch5_Input1x1x2_Kernel32x1x1()
   updateGradInputBatchCUDA(5, 1, 1, 2, 32, 1, 1)
end

function TestUpdateGradInputCUDA:testInput1x2x1_Kernel32x1x1()
   updateGradInputCUDA(1, 2, 1, 32, 1, 1)
end
function TestUpdateGradInputCUDA:testBatch5_Input1x2x1_Kernel32x1x1()
   updateGradInputBatchCUDA(5, 1, 2, 1, 32, 1, 1)
end

function TestUpdateGradInputCUDA:testInput1x1x2_Kernel32x1x2()
   updateGradInputCUDA(1, 1, 2, 32, 1, 2)
end
function TestUpdateGradInputCUDA:testBatch5_Input1x1x2_Kernel32x1x2()
   updateGradInputBatchCUDA(5, 1, 1, 2, 32, 1, 2)
end

function TestUpdateGradInputCUDA:testInput1x2x1_Kernel32x2x1()
   updateGradInputCUDA(1, 2, 1, 32, 2, 1)
end
function TestUpdateGradInputCUDA:testBatch5_Input1x2x1_Kernel32x2x1()
   updateGradInputBatchCUDA(5, 1, 2, 1, 32, 2, 1)
end

function TestUpdateGradInputCUDA:testInput1x64x1_Kernel16x9x1()
   updateGradInputCUDA(1, 64, 1, 16, 9, 1)
end
function TestUpdateGradInputCUDA:testBatch5_Input1x64x1_Kernel16x9x1()
   updateGradInputBatchCUDA(5, 1, 64, 1, 16, 9, 1, 1, 1, 3e-5)
end

function TestUpdateGradInputCUDA:testInput1x1x64_Kernel16x1x9()
   updateGradInputCUDA(1, 1, 64, 16, 1, 9, 1, 1, 6e-5)
end
function TestUpdateGradInputCUDA:testBatch5_Input1x1x64_Kernel16x1x9()
   updateGradInputBatchCUDA(5, 1, 1, 64, 16, 1, 9, 1, 1, 6e-4)
end

function TestUpdateGradInputCUDA:testInput2x1x1_Kernel32x1x1()
   updateGradInputCUDA(2, 1, 1, 32, 1, 1)
end
function TestUpdateGradInputCUDA:testBatch5_Input2x1x1_Kernel32x1x1()
   updateGradInputBatchCUDA(5, 2, 1, 1, 32, 1, 1)
end

function TestUpdateGradInputCUDA:testInput16x1x1_Kernel16x1x1()
   updateGradInputCUDA(16, 1, 1, 16, 1, 1)
end
function TestUpdateGradInputCUDA:testBatch5_Input16x1x1_Kernel16x1x1()
   updateGradInputBatchCUDA(5, 16, 1, 1, 16, 1, 1)
end

function TestUpdateGradInputCUDA:testInput16x11x13_Kernel16x5x7()
   updateGradInputCUDA(16, 11, 13, 16, 5, 7)
end
function TestUpdateGradInputCUDA:testBatch5_Input16x11x13_Kernel16x5x7()
   updateGradInputBatchCUDA(5, 16, 11, 13, 16, 5, 7)
end

function TestUpdateGradInputCUDA:testInput32x11x13_Kernel32x5x7()
   updateGradInputCUDA(32, 11, 13, 32, 5, 7)
end
function TestUpdateGradInputCUDA:testBatch5_Input32x11x13_Kernel32x5x7()
   updateGradInputBatchCUDA(5, 32, 11, 13, 32, 5, 7, 1, 1, 2e-5)
end

-- test that iterating kernel is called for inputs larger than what a
-- single warp can handle that happen to be powers of two.
function TestUpdateGradInputCUDA:testInput64x11x13_Kernel4x5x7()
   updateGradInputCUDA(64, 11, 13, 4, 5, 7)
end
function TestUpdateGradInputCUDA:testBatch5_Input64x11x13_Kernel4x5x7()
   updateGradInputBatchCUDA(5, 64, 11, 13, 4, 5, 7, 1, 1, 2e-5)
end

-- test 256 planes (this is the max supported)
function TestUpdateGradInputCUDA:testInput256x11x13_Kernel256x5x7()
   updateGradInputCUDA(256, 11, 13, 256, 5, 7, 1, 1, 7e-5)
end
function TestUpdateGradInputCUDA:testBatch5_Input256x11x13_Kernel256x5x7()
   updateGradInputBatchCUDA(5, 256, 11, 13, 256, 5, 7, 1, 1, 14e-5)
end


-- -----------------------------------------------------------------------------
-- non-power-of-two tests
-- -----------------------------------------------------------------------------

-- Test on a 5x11x13 input image, 4x5x7 kernel, producing a 4-channel,
-- 7x8 output image.
-- Test that the iterating kernel variant works with small, non-power-of-two
-- input planes and small number of output planes.
function TestUpdateGradInputCUDA:testInput5x11x13_Planes4_Kernel5x7()
   updateGradInputCUDA(5, 11, 13, 4, 5, 7)
end
function TestUpdateGradInputCUDA:testBatch5_Input5x11x13_Planes4_Kernelx5x7()
   updateGradInputBatchCUDA(5, 5, 11, 13, 4, 5, 7)
end

-- test on a 24x11x13 input image, 16x5x7 kernel, producing a 16-channel,
-- 7x8 output image.
-- Test that the iterating kernel variant works with arbitrary numbers of
-- input planes < 32.
function TestUpdateGradInputCUDA:testInput24x11x13_Planes16_Kernel5x7()
   updateGradInputCUDA(24, 11, 13, 16, 5, 7)
end
function TestUpdateGradInputCUDA:testBatch5_Input24x11x13_Planes16_Kernelx5x7()
   updateGradInputBatchCUDA(5, 24, 11, 13, 16, 5, 7, 1, 1, 7e-5)
end

-- test on a 32x11x13 input image, 39x5x7 kernel, producing a 39-channel,
-- 7x8 output image.
-- Test that the iterating kernel variant works with arbitrary numbers of
-- input planes > 32 ("gradOutput" being the input).
function TestUpdateGradInputCUDA:testInput32x11x13_Planes39_Kernel5x7()
   updateGradInputCUDA(32, 11, 13, 39, 5, 7)
end
function TestUpdateGradInputCUDA:testBatch5_Input39x11x13_Planes32_Kernelx5x7()
   updateGradInputBatchCUDA(5, 32, 11, 13, 39, 5, 7, 1, 1, 2e-5)
end

-- test on a 39x11x13 input image, 32x5x7 kernel, producing a 32-channel,
-- 7x8 output image.
-- Test that the iterating kernel variant work with arbitrary numbers of
-- input planes (in particularly not small powers of two or multiples of 32).
function TestUpdateGradInputCUDA:testInput39x11x13_Planes32_Kernel5x7()
   updateGradInputCUDA(39, 11, 13, 32, 5, 7)
end
function TestUpdateGradInputCUDA:testBatch5_Input39x11x13_Planes32_Kernelx5x7()
   updateGradInputBatchCUDA(5, 39, 11, 13, 32, 5, 7, 1, 1, 2e-5)
end


-- -----------------------------------------------------------------------------
-- Facer specific tests
-- -----------------------------------------------------------------------------

-- Facer - local3
function Facer:test_local3_updateGradInput()
   updateGradInputCUDA(16, 64, 64, 16, 9, 9, 1, 1, 5e-5)
end
function Facer:test_local3_updateGradInputBatch()
   updateGradInputBatchCUDA(5, 16, 64, 64, 16, 9, 9, 1, 1, 6e-4)
end

-- Facer - local4
function Facer:test_local4_updateGradInput()
   updateGradInputCUDA(16, 55, 55, 16, 7, 7, 2, 2)
end
function Facer:test_local4_updateGradInputBatch()
   updateGradInputBatchCUDA(5, 16, 55, 55, 16, 7, 7, 2, 2, 1e-4)
end

-- Facer - local5
function Facer:test_local5_updateGradInput()
   updateGradInputCUDA(16, 24, 24, 16, 5, 5)
end
function Facer:test_local5_updateGradInputBatch()
   updateGradInputBatchCUDA(5, 16, 24, 24, 16, 5, 5, 1, 1, 2e-4)
end

local function accGradParametersCUDA(pi, hi, wi, po, hk, wk, scale, hd, wd,
                                     epsW, epsB)
   hd = hd or 1
   wd = wd or 1
   epsW = epsW or 1e-5
   epsB = epsB or 1e-5
   local input, layer = setUpTest(pi, hi, wi, po, hk, wk, hd, wd)
   layer:float()
   local layerCUDA = layer:clone():cuda()

   input = input:float()
   local inputCUDA = input:type('torch.CudaTensor')

   local output = layer:forward(input)
   local outputCUDA = layerCUDA:forward(inputCUDA)
   -- Put an exact copy of the host output into the CUDA tensor.
   -- That way we don't accumulate numerical errors from the forward
   -- pass.
   outputCUDA = output:type('torch.CudaTensor')

   layer:zeroGradParameters()
   layerCUDA:zeroGradParameters()

   layer:accGradParameters(input, output, scale)
   layerCUDA:accGradParameters(inputCUDA, outputCUDA, scale)

   local resultLayerCUDA = layerCUDA:clone():float()
   assertTensorEq(layer.gradWeight, resultLayerCUDA.gradWeight,
                  'LocallyConnected gradWeight CUDA doesn\'t match reference.',
                  epsW)
   assertTensorEq(layer.gradBias, resultLayerCUDA.gradBias,
                  'LocallyConnected gradBias CUDA doesn\'t match reference.',
                  epsB)
end

local function accGradParametersBatchCUDA(bi, pi, hi, wi, po, hk, wk, scale,
                                          hd, wd, epsW, epsB)
   hd = hd or 1
   wd = wd or 1
   epsW = epsW or 1e-5
   epsB = epsB or 1e-5
   local input, layer = setUpBatchTest(bi, pi, hi, wi, po, hk, wk, hd, wd)
   input = input:float()
   layer:float()
   local output = layer:forward(input)
   local inputCUDA = input:type('torch.CudaTensor')
   local layerCUDA = layer:clone():cuda()
   local outputCUDA = layerCUDA:forward(inputCUDA)
   -- Put an exact copy of the host output into the CUDA tensor.
   outputCUDA = output:type('torch.CudaTensor')

   layer:zeroGradParameters()
   layer:accGradParameters(input, output, scale)

   layerCUDA:zeroGradParameters()
   layerCUDA:accGradParameters(inputCUDA, outputCUDA, scale)
   local resultLayerCUDA = layerCUDA:clone():float()
   assertTensorEq(layer.gradWeight, resultLayerCUDA.gradWeight,
                  'LocallyConnected gradWeight CUDA doesn\'t match reference.',
                  epsW)
   assertTensorEq(layer.gradBias, resultLayerCUDA.gradBias,
                  'LocallyConnected gradBias CUDA doesn\'t match reference.',
                  epsB)
end


TestAccGradParametersCUDA = {}

function TestAccGradParametersCUDA:testInput1x1x1_Kernel32x1x1()
   accGradParametersCUDA(1, 1, 1, 32, 1, 1, 0.123)
end
function TestAccGradParametersCUDA:testBatch5_Input1x1x1_Kernel32x1x1()
   accGradParametersBatchCUDA(5, 1, 1, 1, 32, 1, 1, 0.123)
end

function TestAccGradParametersCUDA:testInput1x1x2_Kernel32x1x1()
   accGradParametersCUDA(1, 1, 2, 32, 1, 1, 0.123)
end
function TestAccGradParametersCUDA:testBatch5_Input1x1x2_Kernel32x1x1()
   accGradParametersBatchCUDA(5, 1, 1, 2, 32, 1, 1, 0.123, 1, 1, 2e-7, 5e-7)
end

function TestAccGradParametersCUDA:testInput1x2x1_Kernel32x1x1()
   accGradParametersCUDA(1, 2, 1, 32, 1, 1, 0.123)
end
function TestAccGradParametersCUDA:testBatch5_Input1x2x1_Kernel32x1x1()
   accGradParametersBatchCUDA(5, 1, 2, 1, 32, 1, 1, 0.123)
end

function TestAccGradParametersCUDA:testInput1x1x2_Kernel32x1x2()
   accGradParametersCUDA(1, 1, 2, 32, 1, 2, 0.321)
end
function TestAccGradParametersCUDA:testBatch5_Input1x1x2_Kernel32x1x2()
   accGradParametersBatchCUDA(5, 1, 1, 2, 32, 1, 2, 0.321)
end

function TestAccGradParametersCUDA:testInput1x64x1_Kernel16x9x1()
   accGradParametersCUDA(1, 64, 1, 16, 9, 1, 0.321, 1, 1, 2e-6, 4e-6)
end
function TestAccGradParametersCUDA:testBatch5_Input1x64x1_Kernel16x9x1()
   accGradParametersBatchCUDA(5, 1, 64, 1, 16, 9, 1, 0.321, 1, 1, 1e-5, 3e-5)
end

function TestAccGradParametersCUDA:testInput1x1x64_Kernel16x1x9()
   accGradParametersCUDA(1, 1, 64, 16, 1, 9, 0.321, 1, 1, 5e-7, 4e-6)
end
function TestAccGradParametersCUDA:testBatch5_Input1x1x64_Kernel16x1x9()
   accGradParametersBatchCUDA(5, 1, 1, 64, 16, 1, 9, 0.321, 1, 1, 2e-6, 2e-5)
end

function TestAccGradParametersCUDA:testInput16x11x13_Kernel16x5x7()
   accGradParametersCUDA(16, 11, 13, 16, 5, 7, 0.321, 1, 1, 2e-5, 1e-6)
end
function TestAccGradParametersCUDA:testBatch5_Input16x11x13_Kernel16x5x7()
   accGradParametersBatchCUDA(5, 16, 11, 13, 16, 5, 7, 0.321, 1, 1, 2e-5, 4e-6)
end

function TestAccGradParametersCUDA:testInput32x11x13_Kernel32x5x7()
   accGradParametersCUDA(32, 11, 13, 32, 5, 7, 0.321, 1, 1, 5e-6, 2e-6)
end
function TestAccGradParametersCUDA:testBatch5_Input32x11x13_Kernel32x5x7()
   accGradParametersBatchCUDA(5, 32, 11, 13, 32, 5, 7, 0.321, 1, 1, 3e-5, 5e-6)
end

-- test that iterating kernel is called for inputs larger than what a
-- single warp can handle that happen to be powers of two.
function TestAccGradParametersCUDA:testInput64x11x13_Kernel4x5x7()
   accGradParametersCUDA(64, 11, 13, 4, 5, 7, 0.321, 1, 1, 1e-5, 5e-7)
end
function TestAccGradParametersCUDA:testBatch5_Input64x11x13_Kernel4x5x7()
   accGradParametersBatchCUDA(5, 64, 11, 13, 4, 5, 7, 0.321, 1, 1, 2e-5, 3e-6)
end

-- test with 256 planes, the max supported.
function TestAccGradParametersCUDA:testInput256x11x13_Kernel256x5x7()
   accGradParametersCUDA(256, 11, 13, 256, 5, 7, 0.321, 1, 1, 3.5e-5, 2e-3)
end
function TestAccGradParametersCUDA:testBatch5_Input256x11x13_Kernel256x5x7()
   accGradParametersBatchCUDA(5, 256, 11, 13, 256, 5, 7, 0.321, 1, 1, 2e-4, 0.2)
end


-- -----------------------------------------------------------------------------
-- non-power-of-two tests
-- -----------------------------------------------------------------------------

-- test on a 24x11x13 input image, 16x5x7 kernel, producing a 16-channel,
-- 7x8 output image.
function TestAccGradParametersCUDA:testInput24x11x13_Kernel16x5x7()
   accGradParametersCUDA(24, 11, 13, 16, 5, 7, 0.321, 1, 1, 3e-6, 1e-6)
end
function TestAccGradParametersCUDA:testBatch5_Input24x11x13_Kernel16x5x7()
   accGradParametersBatchCUDA(5, 24, 11, 13, 16, 5, 7, 0.321, 1, 1, 3e-5, 3e-6)
end

-- test on a 32x11x13 input image, 39x5x7 kernel, producing a 39-channel,
-- 7x8 output image.
function TestAccGradParametersCUDA:testInput32x11x13_Kernel39x5x7()
   accGradParametersCUDA(32, 11, 13, 39, 5, 7, 0.321, 1, 1, 5e-6, 2e-6)
end
function TestAccGradParametersCUDA:testBatch5_Input32x11x13_Kernel39x5x7()
   accGradParametersBatchCUDA(5, 32, 11, 13, 39, 5, 7, 0.321, 1, 1, 3e-5, 4e-6)
end

-- test on a 39x11x13 input image, 32x5x7 kernel, producing a 32-channel,
-- 7x8 output image.
-- Test that the iterating kernel variant work with arbitrary numbers of
-- input planes (in particularly not small powers of two or multiples of 32).
function TestAccGradParametersCUDA:testInput39x11x13_Kernel32x5x7()
   accGradParametersCUDA(39, 11, 13, 32, 5, 7, 0.321, 1, 1, 5e-6, 2e-6)
end
function TestAccGradParametersCUDA:testBatch5_Input39x11x13_Kernel32x5x7()
   accGradParametersBatchCUDA(5, 39, 11, 13, 32, 5, 7, 0.321, 1, 1, 3e-5, 4e-6)
end


-- -----------------------------------------------------------------------------
-- Facer specific tests
-- -----------------------------------------------------------------------------

function Facer:test_local3_accGradParameters()
   accGradParametersCUDA(16, 64, 64, 16, 9, 9, 0.1, 1, 1, 2e-5, 8e-4)
end
function Facer:test_local3_accGradParametersBatch()
   accGradParametersBatchCUDA(5, 16, 64, 64, 16, 9, 9, 0.1, 1, 1, 2e-3, 0.2)
end

function Facer:test_local4_accGradParameters()
   accGradParametersCUDA(16, 55, 55, 16, 7, 7, 0.1, 2, 2, 1e-5, 6e-5)
end

function Facer:test_local4_accGradParameterBatch()
   accGradParametersBatchCUDA(5, 16, 55, 55, 16, 7, 7, 0.1, 2, 2, 6e-4, 0.02)
end

function Facer:test_local5_accGradParameters()
   accGradParametersCUDA(16, 24, 24, 16, 5, 5, 0.1, 1, 1, 1e-5, 4e-5)
end
function Facer:test_local5_accGradParametersBatch()
      accGradParametersBatchCUDA(5, 16, 24, 24, 16, 5, 5, 0.1, 1, 1, 3e-4, 9e-3)
end

local function format_time(time, batch_size)
   return string.format('%1.5E / %u = %1.5E', time, batch_size, time/batch_size)
end


-- -----------------------------------------------------------------------------
-- Performance
-- -----------------------------------------------------------------------------

PerformanceCPU = {}

function PerformanceCPU:tearDown()
   collectgarbage()
end

PerformanceGPU = {}

function PerformanceGPU:tearDown()
   collectgarbage()
end

PerformanceCCN = {}

function PerformanceCCN:tearDown()
   collectgarbage()
end

local function updateOutputPerformanceCPU(batch_size)
   local inputCPU, layerCPU = setUpBatchTest(batch_size, 16, 64, 64, 16,
                                             9, 9)
   inputCPU = inputCPU:float()
   layerCPU:float()

   layerCPU:forward(inputCPU)
   local timer = torch.Timer()
   layerCPU:updateOutput(inputCPU)
   local time = timer:time().real
   print(format_time(time, batch_size))
   collectgarbage()
end

local function updateOutputPerformanceGPU(batch_size, planes, inputSize,
                                          kernelSize, stride)
   -- default parameter values
   planes     = planes     or 16
   inputSize  = inputSize  or 64
   kernelSize = kernelSize or 9
   stride     = stride     or 1

   local inputCPU, layerCPU = setUpBatchTest(batch_size, planes, inputSize,
                                             inputSize, planes, kernelSize,
                                             kernelSize, stride, stride)
   local inputGPU = inputCPU:float():cuda()

   -- transpose input's memory layout to CUDA format
   inputGPU = nn.LocallyConnected.toInterleaved(inputGPU, true)
   inputGPU = nn.LocallyConnected.toPlanar(inputGPU)

   layerCPU:float()
   local layerGPU = layerCPU:clone()
   layerGPU:cuda()

   layerGPU:updateOutput(inputGPU)
   cutorch.synchronize()
   local timer = torch.Timer()
   layerGPU:updateOutput(inputGPU)
   cutorch.synchronize()
   time = timer:time().real
   print(format_time(time, batch_size))
   collectgarbage()
end

local function updateOutputPerformanceCCN(batch_size, planes, inputSize,
                                          kernelSize, stride)
   -- default parameter values
   planes     = planes     or 16
   inputSize  = inputSize  or 64
   kernelSize = kernelSize or 9
   stride     = stride     or 1

   local layerCCN = ccn2.SpatialConvolutionLocal(planes, planes, inputSize,
                                                 kernelSize, stride):cuda()
   local inputCCN = torch.Tensor(planes, inputSize, inputSize,
                                 batch_size):uniform(0, 1):cuda()
   layerCCN:updateOutput(inputCCN)
   cutorch.synchronize()
   local timer = torch.Timer()
   layerCCN:updateOutput(inputCCN)
   cutorch.synchronize()
   local time = timer:time().real
   print(format_time(time, batch_size))
   collectgarbage()
end


-- -----------------------------------------------------------------------------
-- updateOutput
-- -----------------------------------------------------------------------------

function PerformanceCPU:testUpdateOutputPerfBatch01()
   updateOutputPerformanceCPU(1)
end

function PerformanceCPU:testUpdateOutputPerfBatch02()
   updateOutputPerformanceCPU(2)
end

function PerformanceCPU:testUpdateOutputPerfBatch04()
   updateOutputPerformanceCPU(4)
end

function PerformanceCPU:testUpdateOutputPerfBatch08()
   updateOutputPerformanceCPU(8)
end

function PerformanceCPU:testUpdateOutputPerfBatch16()
   updateOutputPerformanceCPU(16)
end

function PerformanceGPU:testUpdateOutputPerfBatch001()
   updateOutputPerformanceGPU(1)
end

function PerformanceGPU:testUpdateOutputPerfBatch002()
   updateOutputPerformanceGPU(2)
end

function PerformanceGPU:testUpdateOutputPerfBatch004()
   updateOutputPerformanceGPU(4)
end

function PerformanceGPU:testUpdateOutputPerfBatch008()
   updateOutputPerformanceGPU(8)
end

function PerformanceGPU:testUpdateOutputPerfBatch016()
   updateOutputPerformanceGPU(16)
end

function PerformanceGPU:testUpdateOutputPerfBatch032()
   updateOutputPerformanceGPU(32)
end

function PerformanceGPU:testUpdateOutputPerfBatch064()
   updateOutputPerformanceGPU(64)
end

function PerformanceGPU:testUpdateOutputPerfBatch128()
   updateOutputPerformanceGPU(128)
end

function PerformanceCCN:testUpdateOutputPerfBatch032()
   updateOutputPerformanceCCN(32)
end

function PerformanceCCN:testUpdateOutputPerfBatch064()
   updateOutputPerformanceCCN(64)
end

function PerformanceCCN:testUpdateOutputPerfBatch128()
   updateOutputPerformanceCCN(128)
end

-- -----------------------------------------------------------------------------
-- updateInputGrad
-- -----------------------------------------------------------------------------

local function updateInputGradPerformanceCPU(batch_size)
   local inputCPU, layerCPU = setUpBatchTest(batch_size, 32, 64, 64, 32,
                                             9, 9)
   inputCPU = inputCPU:float()
   layerCPU:float()

   local outputCPU = layerCPU:forward(inputCPU)
   layerCPU:updateGradInput(inputCPU, outputCPU)
   local timer = torch.Timer()
   layerCPU:updateGradInput(inputCPU, outputCPU)
   local time = timer:time().real
   print(format_time(time, batch_size))
   collectgarbage()
end

local function updateInputGradPerformanceGPU(batch_size, planes, inputSize,
                                             kernelSize, stride)
   -- default parameter values
   planes     = planes     or 16
   inputSize  = inputSize  or 64
   kernelSize = kernelSize or 9
   stride     = stride     or 1

   local inputCPU, layerCPU = setUpBatchTest(batch_size, planes, inputSize,
                                             inputSize, planes, kernelSize,
                                             kernelSize, stride, stride)
   local inputGPU = inputCPU:float():cuda()
   layerCPU:float()
   local layerGPU = layerCPU:clone()
   layerGPU:cuda()

   -- transpose input's memory layout to CUDA format
   inputGPU = nn.LocallyConnected.toInterleaved(inputGPU, true)
   inputGPU = nn.LocallyConnected.toPlanar(inputGPU)

   local outputGPU = layerGPU:forward(inputGPU)
   layerGPU:updateGradInput(inputGPU, outputGPU)
   cutorch.synchronize()
   local timer = torch.Timer()
   layerGPU:updateGradInput(inputGPU, outputGPU)
   cutorch.synchronize()
   time = timer:time().real
   print(format_time(time, batch_size))
   collectgarbage()
end

local function updateInputGradPerformanceCCN(batch_size, planes, inputSize,
                                             kernelSize, stride)
   -- default parameter values
   planes     = planes     or 16
   inputSize  = inputSize  or 64
   kernelSize = kernelSize or 9
   stride     = stride     or 1


   local layerCCN = ccn2.SpatialConvolutionLocal(planes, planes, inputSize,
                                                 kernelSize, stride):cuda()
   local inputCCN = torch.Tensor(planes, inputSize, inputSize,
                                 batch_size):uniform(0, 1):cuda()
   local outputCCN = layerCCN:updateOutput(inputCCN)
   layerCCN:updateGradInput(inputCCN, outputCCN)
   cutorch.synchronize()
   local timer = torch.Timer()
   layerCCN:updateGradInput(inputCCN, outputCCN)
   cutorch.synchronize()
   local time = timer:time().real
   print(format_time(time, batch_size))
   collectgarbage()
end

function PerformanceCPU:testUpdateInputGradPerfBatch01()
   updateInputGradPerformanceCPU(1)
end

function PerformanceCPU:testUpdateInputGradPerfBatch02()
   updateInputGradPerformanceCPU(2)
end

function PerformanceCPU:testUpdateInputGradPerfBatch04()
   updateInputGradPerformanceCPU(4)
end

function PerformanceCPU:testUpdateInputGradPerfBatch08()
   updateInputGradPerformanceCPU(8)
end

function PerformanceCPU:testUpdateInputGradPerfBatch16()
   updateInputGradPerformanceCPU(16)
end

function PerformanceGPU:testUpdateInputGradPerfBatch001()
   updateInputGradPerformanceGPU(1)
end

function PerformanceGPU:testUpdateInputGradPerfBatch002()
   updateInputGradPerformanceGPU(2)
end

function PerformanceGPU:testUpdateInputGradPerfBatch004()
   updateInputGradPerformanceGPU(4)
end

function PerformanceGPU:testUpdateInputGradPerfBatch008()
   updateInputGradPerformanceGPU(8)
end

function PerformanceGPU:testUpdateInputGradPerfBatch016()
   updateInputGradPerformanceGPU(16)
end

function PerformanceGPU:testUpdateInputGradPerfBatch032()
   updateInputGradPerformanceGPU(32)
end

function PerformanceGPU:testUpdateInputGradPerfBatch064()
   updateInputGradPerformanceGPU(64)
end

function PerformanceGPU:testUpdateInputGradPerfBatch128()
   updateInputGradPerformanceGPU(128)
end

function PerformanceCCN:testUpdateInputGradPerfBatch032()
   updateInputGradPerformanceCCN(32)
end

function PerformanceCCN:testUpdateInputGradPerfBatch064()
   updateInputGradPerformanceCCN(64)
end

function PerformanceCCN:testUpdateInputGradPerfBatch128()
   updateInputGradPerformanceCCN(128)
end

-- -----------------------------------------------------------------------------
-- accGradParameters
-- -----------------------------------------------------------------------------

local function accGradParametersPerfCPU(batch_size)
   local inputCPU, layerCPU = setUpBatchTest(batch_size, 16, 64, 64, 16,
                                                 9, 9)
   local scale = 0.123
   inputCPU = inputCPU:float()
   layerCPU:float()
   local outputCPU = layerCPU:forward(inputCPU)
   outputCPU:apply(function(x) return torch.uniform() end)

   layerCPU:zeroGradParameters()
   local timer = torch.Timer()
   layerCPU:accGradParameters(inputCPU, outputCPU, scale)
   local time = timer:time().real
   print(format_time(time, batch_size))
   collectgarbage()
end

local function accGradParametersPerfGPU(batch_size, planes, inputSize,
                                        kernelSize, stride)
   -- default parameter values
   planes     = planes     or 16
   inputSize  = inputSize  or 64
   kernelSize = kernelSize or 9
   stride     = stride     or 1

   local inputCPU, layerCPU = setUpBatchTest(batch_size, planes, inputSize,
                                             inputSize, planes, kernelSize,
                                             kernelSize, stride, stride)
   local scale = 0.123
   local inputGPU = inputCPU:float():cuda()
   local layerGPU = layerCPU:clone():cuda()

   -- transpose input's memory layout to CUDA format
   inputGPU = nn.LocallyConnected.toInterleaved(inputGPU, true)
   inputGPU = nn.LocallyConnected.toPlanar(inputGPU)

   local outputGPU = layerGPU:updateOutput(inputGPU)

   layerGPU:zeroGradParameters()
   layerGPU:accGradParameters(inputGPU, outputGPU, scale)
   cutorch.synchronize()
   local timer = torch.Timer()
   layerGPU:accGradParameters(inputGPU, outputGPU, scale)
   cutorch.synchronize()
   local time = timer:time().real
   print(format_time(time, batch_size))
   collectgarbage()
end

local function accGradParametersPerfCCN(batch_size, planes, inputSize,
                                        kernelSize, stride)
   -- default parameter values
   planes     = planes     or 16
   inputSize  = inputSize  or 64
   kernelSize = kernelSize or 9
   stride     = stride     or 1

   local scale      = 0.123
   local layerCCN = ccn2.SpatialConvolutionLocal(planes, planes, inputSize,
                                                 kernelSize, stride):cuda()
   local inputCCN = torch.Tensor(planes, inputSize, inputSize,
                                 batch_size):uniform(0, 1):cuda()
   local outputCCN = layerCCN:updateOutput(inputCCN)

   layerCCN:accGradParameters(inputCCN, outputCCN, scale)
   cutorch.synchronize()
   local timer = torch.Timer()
   layerCCN:accGradParameters(inputCCN, outputCCN, scale)
   cutorch.synchronize()
   local time = timer:time().real
   print(format_time(time, batch_size))
   collectgarbage()
end

function PerformanceCPU:testAccGradParametersPerfBatch01()
   accGradParametersPerfCPU(1)
end

function PerformanceCPU:testAccGradParametersPerfBatch02()
   accGradParametersPerfCPU(2)
end

function PerformanceCPU:testAccGradParametersPerfBatch04()
   accGradParametersPerfCPU(4)
end

function PerformanceCPU:testAccGradParametersPerfBatch08()
   accGradParametersPerfCPU(8)
end

function PerformanceCPU:testAccGradParametersPerfBatch16()
   accGradParametersPerfCPU(16)
end

function PerformanceGPU:testAccGradParametersPerfBatch001()
   accGradParametersPerfGPU(1)
end

function PerformanceGPU:testAccGradParametersPerfBatch002()
   accGradParametersPerfGPU(2)
end

function PerformanceGPU:testAccGradParametersPerfBatch004()
   accGradParametersPerfGPU(4)
end

function PerformanceGPU:testAccGradParametersPerfBatch008()
   accGradParametersPerfGPU(8)
end

function PerformanceGPU:testAccGradParametersPerfBatch016()
   accGradParametersPerfGPU(16)
end

function PerformanceGPU:testAccGradParametersPerfBatch032()
   accGradParametersPerfGPU(32)
end

function PerformanceGPU:testAccGradParametersPerfBatch064()
   accGradParametersPerfGPU(64)
end

function PerformanceGPU:testAccGradParametersPerfBatch128()
   accGradParametersPerfGPU(128)
end

function PerformanceCCN:testAccGradParametersPerfBatch032()
   accGradParametersPerfCCN(32)
end

function PerformanceCCN:testAccGradParametersPerfBatch064()
   accGradParametersPerfCCN(64)
end

function PerformanceCCN:testAccGradParametersPerfBatch128()
   accGradParametersPerfCCN(128)
end

-- -----------------------------------------------------------------------------
-- Facer specific benchmark
-- -----------------------------------------------------------------------------

FacerPerformance = {}

function FacerPerformance:tearDown()
   collectgarbage()
end

local function newLocal(inputSize, stride, planes, kernel)
   print('   Input: ' .. planes .. 'x' .. inputSize .. 'x' .. inputSize)
   print('   Kernel: ' .. planes .. 'x' .. kernel .. 'x' .. kernel)
   print('   Stride: ' .. stride)
   print('Torch: ')
   print('updateOutput: ')
   updateOutputPerformanceGPU(32, planes, inputSize, kernel, stride)
   updateOutputPerformanceGPU(64, planes, inputSize, kernel, stride)
   updateOutputPerformanceGPU(96, planes, inputSize, kernel, stride)
   updateOutputPerformanceGPU(128, planes, inputSize, kernel, stride)
   print('updateGradInput: ')
   updateInputGradPerformanceGPU(32, planes, inputSize, kernel, stride)
   updateInputGradPerformanceGPU(64, planes, inputSize, kernel, stride)
   updateInputGradPerformanceGPU(96, planes, inputSize, kernel, stride)
   updateInputGradPerformanceGPU(128, planes, inputSize, kernel, stride)
   print('accGradParameters: ')
   accGradParametersPerfGPU(32, planes, inputSize, kernel, stride)
   accGradParametersPerfGPU(64, planes, inputSize, kernel, stride)
   accGradParametersPerfGPU(96, planes, inputSize, kernel, stride)
   accGradParametersPerfGPU(128, planes, inputSize, kernel, stride)
   print('ConvNet2: ')
   print('updateOutput: ')
   updateOutputPerformanceCCN(32, planes, inputSize, kernel, stride)
   updateOutputPerformanceCCN(64, planes, inputSize, kernel, stride)
   updateOutputPerformanceCCN(96, planes, inputSize, kernel, stride)
   updateOutputPerformanceCCN(128, planes, inputSize, kernel, stride)
   print('updateGradInput: ')
   if (planes == 16) then
      print('n/a / 32 = n/a')
      print('n/a / 64 = n/a')
      print('n/a / 96 = n/a')
      print('n/a / 128 = n/a')
   else
      updateInputGradPerformanceCCN(32, planes, inputSize, kernel, stride)
      updateInputGradPerformanceCCN(64, planes, inputSize, kernel, stride)
      updateInputGradPerformanceCCN(96, planes, inputSize, kernel, stride)
      updateInputGradPerformanceCCN(128, planes, inputSize, kernel, stride)
   end
   print('accGradParameters: ')
   accGradParametersPerfCCN(32, planes, inputSize, kernel, stride)
   accGradParametersPerfCCN(64, planes, inputSize, kernel, stride)
   accGradParametersPerfCCN(96, planes, inputSize, kernel, stride)
   accGradParametersPerfCCN(128, planes, inputSize, kernel, stride)
end

function FacerPerformance:testNewLocal3()
   newLocal(44, 1, 32, 3)
   newLocal(152, 1, 32, 3)
end

function FacerPerformance:testNewLocal4()
   newLocal(42, 2, 32, 3)
   newLocal(150, 2, 32, 3)
end

function FacerPerformance:testLocal3()
   newLocal(63, 1, 16, 9)
end

function FacerPerformance:testLocal4()
   newLocal(57, 2, 16, 7)
end

function FacerPerformance:testLocal5()
   newLocal(26, 1, 16, 5)
end

function FacerPerformance:testKernel9()
   newLocal(50, 1, 32, 9)
end


LuaUnit:main()

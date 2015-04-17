-- Copyright 2004-present Facebook. All Rights Reserved.

require('fb.luaunit')
require('fbtorch')
require('nn')
require('cutorch')
require('cunn')

local jac = nn.Jacobian

local debugger = require('fb.debugger')

local function assertTensorEq(a, b, msg, precision)
   precision = precision or 1e-5
   local diff = torch.dist(a, b)
   if diff > precision then
      debugger:enter()
      error('diff = ' .. diff .. ': ' .. msg)
   end
end

TestParameterValidation = {}

function TestParameterValidation:tearDown()
   collectgarbage()
   end

-- Check that constructor parameters get propagated correctly.
--
function TestParameterValidation:testConstructor()
   local layer = nn.VolumetricAveragePooling(3, 5, 7)
   assertEquals(3, layer.kT)
   assertEquals(5, layer.kW)
   assertEquals(7, layer.kH)

   layer = nn.VolumetricAveragePooling(3, 5, 7, 2, 4, 6)
   assertEquals(2, layer.dT)
   assertEquals(4, layer.dW)
   assertEquals(6, layer.dH)
end

-- Check that input tensor dimension is validated during forward().
--
function TestParameterValidation:UpdateOutputDimensions(type)
   local layer = nn.VolumetricAveragePooling(3, 5, 7):type(type)

   local input = torch.Tensor(5, 9, 5):type(type) -- 3D Tensor: bad
   local status = pcall(function() layer:updateOutput(input) end)
   assertEquals(false, status, "Invalid input size not detected.")

   input = torch.Tensor(1, 3, 5, 7, 11, 9):type(type) -- 6D Tensor: bad, too
   status = pcall(function() layer:updateOutput(input) end)
   assertEquals(false, status, "Invalid input size not detected.")

   input = torch.Tensor(1, 5, 9, 7):type(type) -- 4D Tensor, proper size: good
   status = pcall(function() layer:updateOutput(input) end)
   assertEquals(true, status, "Valid input caused error.")

   input = torch.Tensor(1, 3, 5, 9, 7):type(type)
   status = pcall(function() layer:updateOutput(input) end)
   assertEquals(true, status, "Valid input caused error.")
end

function TestParameterValidation:testUpdateOutputDimensionsCPU()
   self:UpdateOutputDimensions('torch.DoubleTensor')
end

function TestParameterValidation:testUpdateOutputDimensionsGPU()
   self:UpdateOutputDimensions('torch.CudaTensor')
end

-- Check that input tensor size is properly validated.
--
function TestParameterValidation:UpdateOutputInputSize(type)
   local layer = nn.VolumetricAveragePooling(3, 5, 7):type(type)

   -- the following is the smallest valid input to the above
   -- layer
   local input = torch.Tensor(1, 3, 7, 5):type(type)
   local status = pcall(function() layer:updateOutput(input) end)
   assertEquals(true, status, "Invalid input width not detected.")

   input = torch.Tensor(1, 2, 7, 5):type(type)
   status = pcall(function() layer:updateOutput(input) end)
   assertEquals(false, status, "Invalid input time slices not detected.")

   input = torch.Tensor(10, 3, 6, 5):type(type)
   status = pcall(function() layer:updateOutput(input) end)
   assertEquals(false, status, "Invalid input height not detected.")

   input = torch.Tensor(1, 3, 7, 4):type(type)
   status = pcall(function() layer:updateOutput(input) end)
   assertEquals(false, status, "Invalid input width not detected.")
end

function TestParameterValidation:testUpdateOutputInputSizeCPU()
   self:UpdateOutputInputSize('torch.DoubleTensor')
end

function TestParameterValidation:testUpdateOutputInputSizeGPU()
   self:UpdateOutputInputSize('torch.CudaTensor')
end

-- Checkt that output tensor is properly sized.
--
function TestParameterValidation:UpdateOutputOutputSize(type)
   local layer = nn.VolumetricAveragePooling(3, 5, 7, 1, 1, 1):type(type)
   local input = torch.Tensor(1, 3, 7, 5):type(type);

   local output = layer:updateOutput(input)

   assertEquals(input:size(1), output:size(1),
                "Wrong number of output features.")
   assertEquals((input:size(2) - layer.kT) / layer.dT + 1, output:size(2),
      "Wrong number of output frames.")
   assertEquals((input:size(3) - layer.kH) / layer.dH + 1, output:size(3),
      "Wrong output height.")
   assertEquals((input:size(4) - layer.kW) / layer.dW + 1, output:size(4),
      "Wrong output width.")
end

function TestParameterValidation:UpdateOutputOutputSizeBatch(type)
   local layer = nn.VolumetricAveragePooling(3, 5, 7, 1, 1, 1):type(type)
   local input = torch.Tensor(4, 1, 3, 7, 5):type(type);

   local output = layer:updateOutput(input)

   assertEquals(input:size(1), output:size(1),
                "Wrong batch size.")
   assertEquals(input:size(2), output:size(2),
                "Wrong number of output features.")
   assertEquals((input:size(3) - layer.kT) / layer.dT + 1, output:size(3),
      "Wrong number of output frames.")
   assertEquals((input:size(4) - layer.kH) / layer.dH + 1, output:size(4),
      "Wrong output height.")
   assertEquals((input:size(5) - layer.kW) / layer.dW + 1, output:size(5),
      "Wrong output width.")
end

function TestParameterValidation:testUpdateOutputOutputSizeCPU()
   self:UpdateOutputOutputSize('torch.DoubleTensor')
end

function TestParameterValidation:testUpdateOutputOutputSizeGPU()
   self:UpdateOutputOutputSize('torch.CudaTensor')
end

function TestParameterValidation:testUpdateOutputOutputSizeBatchCPU()
   self:UpdateOutputOutputSizeBatch('torch.DoubleTensor')
end

function TestParameterValidation:testUpdateOutputOutputSizeBatchGPU()
   self:UpdateOutputOutputSizeBatch('torch.CudaTensor')
end


TestHandGilded = {}

function TestHandGilded:testUpdateOutput01()
   local layer = nn.VolumetricAveragePooling(2, 2, 2)
   local input = torch.Tensor(1, 2, 2, 2):fill(1.0)
   local output = layer:updateOutput(input)
   assertEquals(1, output[1][1][1][1], 'Incorrect output')
end

function TestHandGilded:testUpdateOutput02()
   local layer = nn.VolumetricAveragePooling(1, 1, 1)
   local input = torch.Tensor(1, 10, 11, 12):uniform(0.0, 1.0)
   local output = layer:updateOutput(input)
   assertTensorEq(input, output, 'Input must match output')
end


TestJacobian = {}

function TestJacobian:testUpdateGradInputJacobian01()
   local layer = nn.VolumetricAveragePooling(2, 2, 2)
   local input = torch.Tensor(1, 2, 2, 2):uniform(0.0, 1.0)
   local err = jac.testJacobian(layer, input)
   assertAlmostEquals(err, 0.0, 1e-7)
end

function TestJacobian:testUpdateGradInputJacobian02()
   local layer = nn.VolumetricAveragePooling(3, 5, 7)
   local input = torch.Tensor(16, 11, 13, 17):uniform(-1.0, 1.0)
   local err = jac.testJacobian(layer, input)
   assertAlmostEquals(err, 0.0, 1e-7)
end


-- -----------------------------------------------------------------------------
-- CUDA implemenation tests - comparing to CPU as reference
-- -----------------------------------------------------------------------------


TestUpdateOutputCUDA = {}

function TestUpdateOutputCUDA:tearDown()
   collectgarbage()
end

function TestUpdateOutputCUDA:updateOutputTest(iF, iT, iH, iW, kT, kH, kW,
                                               dT, dH, dW)

   local input = torch.Tensor(iF, iT, iH, iW):float():uniform(-1, 1)
   local layer = nn.VolumetricAveragePooling(kT, kH, kW, dT, dH, dW):float()

   local inputCUDA = input:cuda()
   local layerCUDA = layer:clone():cuda()

   local output = layer:updateOutput(input)
   local outputCUDA = layerCUDA:updateOutput(inputCUDA)

   local outputHostCUDA = outputCUDA:float()
   assertTensorEq(output, outputHostCUDA,
                  'Cuda output doesn\'t match host reference');
end

function TestUpdateOutputCUDA:test01()
   self:updateOutputTest(3, 5, 7, 11, 3, 5, 7, 1, 1, 1)
end

function TestUpdateOutputCUDA:test02()
   self:updateOutputTest(3, 5, 7,  11, 3, 5, 7, 2, 3, 4)
end

function TestUpdateOutputCUDA:test03()
   self:updateOutputTest(3, 5, 7, 11, 3, 5, 7, 3, 5, 7)
end

function TestUpdateOutputCUDA:test04()
   self:updateOutputTest(1, 8, 8, 8, 5, 5, 5, 3, 3, 3)
end


-- -----------------------------------------------------------------------------

TestUpdateGradInputCUDA = {}

function TestUpdateGradInputCUDA:tearDown()
   collectgarbage()
end

function TestUpdateGradInputCUDA:updateGradInputTest(iF, iT, iH, iW, kT, kH, kW,
                                                     dT, dH, dW)
   local input = torch.Tensor(iF, iT, iH, iW):float():uniform(-1, 1)
   local layer = nn.VolumetricAveragePooling(kT, kH, kW, dT, dH, dW):float()

   local inputCUDA = input:cuda()
   local layerCUDA = layer:clone():cuda()

   local output = layer:updateOutput(input)
   layerCUDA:updateOutput(inputCUDA)

   local gradOutput = output:clone():uniform(-1, 1)
   local gradOutputCUDA = gradOutput:cuda()

   local gradInput = layer:updateGradInput(input, gradOutput)
   local gradInputCUDA = layerCUDA:updateGradInput(inputCUDA, gradOutputCUDA)

   local gradInputHostCUDA = gradInputCUDA:float()
   assertTensorEq(gradInput, gradInputHostCUDA,
                  'Cuda gradInput doesn\'t match host reference');
end

function TestUpdateGradInputCUDA:test01()
   self:updateGradInputTest(3, 5, 7, 11, 3, 5, 7, 1, 1, 1)
end

function TestUpdateGradInputCUDA:test02()
   self:updateGradInputTest(3, 5, 7, 11, 3, 5, 7, 2, 3, 4)
end

function TestUpdateGradInputCUDA:test03()
   self:updateGradInputTest(3, 5, 7, 11, 3, 5, 7, 3, 5, 7)
end

function TestUpdateGradInputCUDA:test04()
   self:updateGradInputTest(1, 8, 8, 8, 5, 5, 5, 3, 3, 3)
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

local function updateOutputPerformanceCPU(pi, ti, hi, wi, tk, hk, wk,
                                          td, hd, wd)
   local input = torch.Tensor(pi, ti, hi, wi):float():uniform(-1, 1)
   local layer = nn.VolumetricAveragePooling(tk, hk, wk, td, hd, wd):float()

   layer:forward(input)
   local timer = torch.Timer()
   layer:updateOutput(input)
   local time = timer:time().real
   print(string.format('time = %1.5E s', time))
end

local function updateOutputPerformanceGPU(pi, ti, hi, wi, tk, hk, wk,
                                          td, hd, wd)
   local input = torch.Tensor(pi, ti, hi, wi):float():uniform(-1, 1)
   local layer = nn.VolumetricAveragePooling(tk, hk, wk, td, hd, wd):float()

   local inputCUDA = input:type('torch.CudaTensor')
   local layerCUDA = layer:clone():cuda()

   layerCUDA:updateOutput(inputCUDA)
   cutorch.synchronize()
   local timer = torch.Timer()
   layerCUDA:updateOutput(inputCUDA)
   cutorch.synchronize()
   local time = timer:time().real
   print(string.format('time = %1.5E s', time))
end


-- -----------------------------------------------------------------------------
-- updateOutput
-- -----------------------------------------------------------------------------

function PerformanceCPU:testUpdateOutputPerf1()
   updateOutputPerformanceCPU(128, 128, 152, 152, 3, 3, 3, 1, 1, 1)
end

function PerformanceCPU:testUpdateOutputPerf2()
   updateOutputPerformanceCPU(128, 128, 152, 152, 3, 3, 3, 2, 2, 2)
end

function PerformanceCPU:testUpdateOutputPerf3()
   updateOutputPerformanceCPU(128, 128, 152, 152, 3, 3, 3, 3, 3, 3)
end

function PerformanceGPU:testUpdateOutputPerf1()
   updateOutputPerformanceGPU(128, 128, 152, 152, 3, 3, 3, 1, 1, 1)
end

function PerformanceGPU:testUpdateOutputPerf2()
   updateOutputPerformanceGPU(128, 128, 152, 152, 3, 3, 3, 2, 2, 2)
end

function PerformanceGPU:testUpdateOutputPerf3()
   updateOutputPerformanceGPU(128, 128, 152, 152, 3, 3, 3, 3, 3, 3)
end


-- -----------------------------------------------------------------------------
-- updateInputGrad
-- -----------------------------------------------------------------------------

local function updateInputGradPerformanceCPU(pi, ti, hi, wi, tk, hk, wk,
                                             td, hd, wd)
   local input = torch.Tensor(pi, ti, hi, wi):float():uniform(-1, 1)
   local layer = nn.VolumetricAveragePooling(tk, hk, wk, td, hd, wd):float()

   layer:forward(input)

   local output = layer:forward(input)
   layer:updateGradInput(input, output)
   local timer = torch.Timer()
   layer:updateGradInput(input, output)
   local time = timer:time().real
   print(string.format('time = %1.5E s', time))
end

local function updateInputGradPerformanceGPU(pi, ti, hi, wi, tk, hk, wk,
                                             td, hd, wd)
   local input = torch.Tensor(pi, ti, hi, wi):float():uniform(-1, 1)
   local layer = nn.VolumetricAveragePooling(tk, hk, wk, td, hd, wd):float()

   local inputCUDA = input:type('torch.CudaTensor')
   local layerCUDA = layer:clone():cuda()

   layerCUDA:updateOutput(inputCUDA)
   local outputCUDA = layerCUDA:forward(inputCUDA)
   layerCUDA:updateGradInput(inputCUDA, outputCUDA)
   cutorch.synchronize()
   local timer = torch.Timer()
   layerCUDA:updateGradInput(inputCUDA, outputCUDA)
   cutorch.synchronize()
   local time = timer:time().real
   print(string.format('time = %1.5E s', time))
end

function PerformanceCPU:testUpdateGradInputPerf1()
   updateInputGradPerformanceCPU(128, 128, 152, 152, 3, 3, 3, 1, 1, 1)
end

function PerformanceCPU:testUpdateGradInputPerf2()
   updateInputGradPerformanceCPU(128, 128, 152, 152, 3, 3, 3, 2, 2, 2)
end

function PerformanceCPU:testUpdateGradInputPerf3()
   updateInputGradPerformanceCPU(128, 128, 152, 152, 3, 3, 3, 3, 3, 3)
end

function PerformanceGPU:testUpdateGradInputPerf1()
   updateInputGradPerformanceGPU(128, 128, 152, 152, 3, 3, 3, 1, 1, 1)
end

function PerformanceGPU:testUpdateGradInputPerf2()
   updateInputGradPerformanceGPU(128, 128, 152, 152, 3, 3, 3, 2, 2, 2)
end

function PerformanceGPU:testUpdateGradInputPerf3()
   updateInputGradPerformanceGPU(128, 128, 152, 152, 3, 3, 3, 3, 3, 3)
end

LuaUnit:main()

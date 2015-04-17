-- Copyright 2004-present Facebook. All Rights Reserved.

require('fb.luaunit')
require('fbtorch')
require('nn')
require('cutorch')
require('cunn')


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

-- Check that constructor parameters get propagated correctly.
--
function TestParameterValidation:testConstructor()
   local layer = nn.VolumetricMaxPooling(3, 5, 7)
   assertEquals(3, layer.kT)
   assertEquals(5, layer.kW)
   assertEquals(7, layer.kH)

   layer = nn.VolumetricMaxPooling(3, 5, 7, 2, 4, 6)
   assertEquals(2, layer.dT)
   assertEquals(4, layer.dW)
   assertEquals(6, layer.dH)
end

-- Check that input tensor dimension is validated during forward().
--
function TestParameterValidation:UpdateOutputDimensions(type)
   local layer = nn.VolumetricMaxPooling(3, 5, 7):type(type)

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
   local layer = nn.VolumetricMaxPooling(3, 5, 7):type(type)

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
   local layer = nn.VolumetricMaxPooling(3, 5, 7, 1, 1, 1):type(type)
   local input = torch.Tensor(1, 3, 7, 5):type(type);

   local output = layer:updateOutput(input)

   assertEquals(input:size(1), output:size(1), "Wrong number of output features.")
   assertEquals((input:size(2) - layer.kT) / layer.dT + 1, output:size(2),
      "Wrong number of output frames.")
   assertEquals((input:size(3) - layer.kH) / layer.dH + 1, output:size(3),
      "Wrong output height.")
   assertEquals((input:size(4) - layer.kW) / layer.dW + 1, output:size(4),
      "Wrong output width.")
end

function TestParameterValidation:UpdateOutputOutputSizeBatch(type)
   local layer = nn.VolumetricMaxPooling(3, 5, 7, 1, 1, 1):type(type)
   local input = torch.Tensor(4, 1, 3, 7, 5):type(type);

   local output = layer:updateOutput(input)

   assertEquals(input:size(1), output:size(1), "Wrong batch size.")
   assertEquals(input:size(2), output:size(2), "Wrong number of output features.")
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


-- -----------------------------------------------------------------------------
-- CUDA implemenation tests - comparing to CPU as reference
-- -----------------------------------------------------------------------------

local function outputCUDA(bi, pi, ti, hi, wi, tk, hk, wk, td, hd, wd)
   td = td or 1
   hd = hd or 1
   wd = wd or 1

   local input
   if (bi == 0) then
      input = torch.Tensor(pi, ti, hi, wi):float():uniform(-1, 1)
   else
      input = torch.Tensor(bi, pi, ti, hi, wi):float():uniform(-1, 1)
   end
   local layer = nn.VolumetricMaxPooling(tk, hk, wk, td, hd, wd):float()

   local inputCUDA = input:type('torch.CudaTensor')
   local layerCUDA = layer:clone():cuda()
   local output = layer:forward(input)
   local outputCUDA = layerCUDA:forward(inputCUDA)
   assert(output:dim() == outputCUDA:dim())
   for dim = 1,output:dim() do
      assert(output:size(dim) == outputCUDA:size(dim), 'Output size mismatch.')
      assert(input:size(dim) == inputCUDA:size(dim), 'Input size mismatch.')
   end
   local outputHostCUDA = outputCUDA:float()
   assertTensorEq(output, outputHostCUDA,
                  'VolumetricMaxPool CUDA output doesn\'t match reference.',
                  0.0)
   local indices = layer.indices
   local indicesHostCUDA = layerCUDA.indices:float()
   assertTensorEq(indices, indicesHostCUDA,
                  'VolumetricMaxPool CUDA indices doesn\'t match reference.',
                  0.0)
end

TestUpdateOutputCUDA = {}

function TestUpdateOutputCUDA:tearDown()
   collectgarbage()
end

local function createTestName(params)
   local testName = 'test'
   if (params[1] ~= 0) then -- batch test
      testName = testName .. 'Batch_' .. params[1] .. '_'
   end
   testName = testName .. 'Input_' .. params[2] .. 'x'
      .. params[3] .. 'x' .. params[4] .. 'x'
      .. params[5] .. '_Kernel_' .. params[6]
      .. 'x' .. params[7] .. 'x' .. params[8]
   if (#params >= 11) then -- stride
      testName = testName .. '_Stride_' .. params[9]
         .. 'x' .. params[10] .. 'x' .. params[11]
   end

   return testName
end

local testParams = {
   {0,  1,   1,   1,   1, 1, 1, 1},
   {0,  1,   2,   2,   2, 2, 2, 2},
   {0,  1,   4,   4,   4, 2, 2, 2, 2, 2, 2},
   {0,  1,   4,   4,   4, 2, 2, 2, 1, 2, 2},
   {0,  1,   4,   4,   4, 2, 2, 2, 2, 1, 2},
   {0,  1,   4,   4,   4, 2, 2, 2, 2, 2, 1},
   {0,  1,   6,   6,   6, 3, 3, 3},
   {0,  1,   6,   6,   6, 3, 3, 3, 2, 2, 2},
   {0,  1,   6,   6,   6, 3, 3, 3, 2, 2, 1},
   {0,  1,   6,   6,   6, 3, 3, 3, 2, 1, 2},
   {0,  1,   6,   6,   6, 3, 3, 3, 1, 2, 2},
   {0,  2,   6,   6,   6, 3, 3, 3, 1, 2, 2},
   {0, 64, 128, 152, 152, 3, 3, 3, 1, 1, 1, 1e-3},
   {0, 64, 128, 152, 152, 3, 3, 3, 2, 2, 2, 1e-4},
   {0, 16,   1,   1,   1, 1, 1, 1},

   -- batched tests
   {16, 1,  1,   1,   1, 1, 1, 1},
   {16, 1,  2,   2,   2, 2, 2, 2},
   {16, 1,  4,   4,   4, 2, 2, 2, 2, 2, 2},
   {16, 1,  4,   4,   4, 2, 2, 2, 1, 2, 2},
   {16, 1,  4,   4,   4, 2, 2, 2, 2, 1, 2},
   {16, 1,  4,   4,   4, 2, 2, 2, 2, 2, 1},
   {16, 1,  6,   6,   6, 3, 3, 3, 1, 1, 1, 1e-5},
   {16, 1,  6,   6,   6, 3, 3, 3, 2, 2, 2},
   {16, 1,  6,   6,   6, 3, 3, 3, 2, 2, 1},
   {16, 1,  6,   6,   6, 3, 3, 3, 2, 1, 2},
   {16, 1,  6,   6,   6, 3, 3, 3, 1, 2, 2},
   {16, 2,  6,   6,   6, 3, 3, 3, 1, 2, 2},
   {16, 4, 32, 128, 128, 3, 3, 3, 1, 1, 1, 1e-3},
   {16, 4, 32, 128, 128, 3, 3, 3, 2, 2, 2, 1e-4},
}

for i, params in ipairs(testParams) do
   TestUpdateOutputCUDA[createTestName(params)] = function()
      outputCUDA(unpack(params))
   end
end

-- -----------------------------------------------------------------------------

local function gradInputCUDA(bi, pi, ti, hi, wi, tk, hk, wk, td, hd, wd, eps)
   td = td or 1
   hd = hd or 1
   wd = wd or 1
   eps = eps or 1e-6

   local input
   if (bi == 0) then
      input = torch.Tensor(pi, ti, hi, wi):float():uniform(-1, 1)
   else
      input = torch.Tensor(bi, pi, ti, hi, wi):float():uniform(-1, 1)
   end
   local layer = nn.VolumetricMaxPooling(tk, hk, wk, td, hd, wd):float()

   local inputCUDA = input:type('torch.CudaTensor')
   local layerCUDA = layer:clone():cuda()
   local output = layer:forward(input)
   local outputCUDA = layerCUDA:forward(inputCUDA)

   local gradOutput = output:clone():uniform(-1, 1)
   local gradOutputCUDA = gradOutput:type('torch.CudaTensor')

   local gradInput = layer:backward(input, gradOutput)
   local gradInputCUDA = layerCUDA:backward(inputCUDA, gradOutputCUDA)

   local gradInputHostCUDA = gradInputCUDA:float()
   assertTensorEq(gradInput, gradInputHostCUDA,
                  'VolumetricMaxPool CUDA gradInput doesn\'t match reference.',
                  eps)
end


TestUpdateGradInputCUDA = {}

function TestUpdateGradInputCUDA:tearDown()
   collectgarbage()
end

for i, params in ipairs(testParams) do
   TestUpdateGradInputCUDA[createTestName(params)] = function()
      gradInputCUDA(unpack(params))
   end
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
   td = td or 1
   hd = hd or 1
   wd = wd or 1

   local input = torch.Tensor(pi, ti, hi, wi):float():uniform(-1, 1)
   local layer = nn.VolumetricMaxPooling(tk, hk, wk, td, hd, wd):float()

   layer:forward(input)
   local timer = torch.Timer()
   layer:updateOutput(input)
   local time = timer:time().real
   print(string.format('time = %1.5E s', time))
end

local function updateOutputPerformanceGPU(pi, ti, hi, wi, tk, hk, wk,
                                          td, hd, wd)
   td = td or 1
   hd = hd or 1
   wd = wd or 1

   local input = torch.Tensor(pi, ti, hi, wi):float():uniform(-1, 1)
   local layer = nn.VolumetricMaxPooling(tk, hk, wk, td, hd, wd):float()

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
   updateOutputPerformanceCPU(128, 128, 152, 152, 3, 3, 3)
end

function PerformanceCPU:testUpdateOutputPerf2()
   updateOutputPerformanceCPU(128, 128, 152, 152, 3, 3, 3, 2, 2, 2)
end

function PerformanceGPU:testUpdateOutputPerf1()
   updateOutputPerformanceGPU(128, 128, 152, 152, 3, 3, 3)
end

function PerformanceGPU:testUpdateOutputPerf2()
   updateOutputPerformanceGPU(128, 128, 152, 152, 3, 3, 3, 2, 2, 2)
end


-- -----------------------------------------------------------------------------
-- updateInputGrad
-- -----------------------------------------------------------------------------

local function updateInputGradPerformanceCPU(pi, ti, hi, wi, tk, hk, wk,
                                             td, hd, wd)
   td = td or 1
   hd = hd or 1
   wd = wd or 1

   local input = torch.Tensor(pi, ti, hi, wi):float():uniform(-1, 1)
   local layer = nn.VolumetricMaxPooling(tk, hk, wk, td, hd, wd):float()

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
   td = td or 1
   hd = hd or 1
   wd = wd or 1

   local input = torch.Tensor(pi, ti, hi, wi):float():uniform(-1, 1)
   local layer = nn.VolumetricMaxPooling(tk, hk, wk, td, hd, wd):float()

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

function PerformanceCPU:testUpdateInputGradPerf1()
   updateInputGradPerformanceCPU(128, 128, 152, 152, 3, 3, 3)
end

function PerformanceCPU:testUpdateInputGradPerf2()
   updateInputGradPerformanceCPU(128, 128, 152, 152, 3, 3, 3, 2, 2, 2)
end

function PerformanceGPU:testUpdateInputGradPerf1()
   updateInputGradPerformanceGPU(128, 128, 152, 152, 3, 3, 3)
end

function PerformanceGPU:testUpdateInputGradPerf2()
   updateInputGradPerformanceGPU(128, 128, 152, 152, 3, 3, 3, 2, 2, 2)
end


LuaUnit:main()

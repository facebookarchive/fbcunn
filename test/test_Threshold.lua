-- Copyright 2004-present Facebook. All Rights Reserved.

require('fb.luaunit')
require('fbtorch')
require('nn')
require('cutorch')
require('cunn')


local function assertTensorEq(a, b, msg, precision)
   precision = precision or 1e-5
   msg = msg or 'error'
   local diff = torch.dist(a, b)
   if diff > precision then
      error('diff = ' .. diff .. ': ' .. msg)
   end

end


TestUpdateOutput = {}

function TestUpdateOutput:setUp()
   self.hostLayer = nn.Threshold(0.0, 0.0):float()
   self.hostInput = torch.Tensor(16, 10, 10):float():uniform(-1.0, 1.0)

   self.hostOutput = self.hostLayer:forward(self.hostInput)

   self.cudaLayer = self.hostLayer:clone():cuda()
   self.cudaInput = self.hostInput:cuda()
end

function TestUpdateOutput:tearDown()
   collectgarbage()
end

function TestUpdateOutput:compareResults()
   local hostCudaOutput = self.cudaOutput:float()

   assertTensorEq(self.hostOutput, hostCudaOutput, "Results don't match", 0.0)
end

function TestUpdateOutput:testBasic()
   self.cudaOutput = self.cudaLayer:forward(self.cudaInput)
   self:compareResults()
end

function TestUpdateOutput:testTransposed()
   -- transpose the cuda input, make contiguous and transpose back
   -- this results in a tensor with original sizes that is not contiguous,
   -- i.e. it is in a transposed state and if iterated in lexicographical
   -- order would not result in a linear motions through address space.
   self.cudaInput = self.cudaInput:transpose(1, 3):contiguous():transpose(3, 1)
   self.cudaOutput = self.cudaLayer:forward(self.cudaInput)
   self:compareResults()
end


TestUpdateOutputInPlace = {}

function TestUpdateOutputInPlace:setUp()
   TestUpdateOutput.setUp(self)
   self.cudaLayer.inplace = true
end

function TestUpdateOutputInPlace:tearDown()
   collectgarbage()
end

function TestUpdateOutputInPlace:compareResults()
   TestUpdateOutput.compareResults(self)
end

function TestUpdateOutputInPlace:testBasic()
   self.cudaOutput = self.cudaLayer:forward(self.cudaInput)
   self:compareResults()
end

function TestUpdateOutputInPlace:testPreserveTransposition()
   -- to avoid unnecessary copies when chaining layers that operate on
   -- transposed tensors (i.e. transparently use a different memory layout from
   -- Torch's default, point-wise layers should preserve the memory layout of
   -- their input
   self.cudaInput = self.cudaInput:transpose(1, 3):contiguous():transpose(3, 1)
   self.cudaOutput = self.cudaLayer:forward(self.cudaInput)

   assert(self.cudaInput:dim() == self.cudaOutput:dim())
   for i = 1, self.cudaInput:dim() do
      assert(self.cudaInput:size(i) == self.cudaOutput:size(i))
      assert(self.cudaInput:stride(i) == self.cudaOutput:stride(i))
   end
end


TestUpdateGradInput = {}

function TestUpdateGradInput:setUp()
   -- use setup from updateOutput() tests
   TestUpdateOutput.setUp(self)
   -- run forward pass cuda
   self.cudaOutput = self.cudaLayer:forward(self.cudaInput)
   -- create gradOutput tensor and run backward pass
   self.hostGradOutput = torch.Tensor(16, 10, 10):float():uniform(-1.0, 1.0)
   self.hostGradInput = self.hostLayer:backward(self.hostInput,
                                                self.hostGradOutput)
   self.cudaGradOutput = self.hostGradOutput:cuda()
end

function TestUpdateGradInput:tearDown()
   collectgarbage()
end

function TestUpdateGradInput:compareResults()
   local hostCudaGradInput = self.cudaGradInput:float()

   assertTensorEq(self.hostGradInput, hostCudaGradInput, "Results don't match",
                  0.0)
end

function TestUpdateGradInput:testBasic()
   self.cudaGradInput = self.cudaLayer:backward(self.cudaInput,
                                                self.cudaGradOutput)
   self:compareResults()
end

function TestUpdateGradInput:testTransposed()
   -- this test puts the gradOutput into a transposed state
   self.cudaGradOutput = self.cudaGradOutput:transpose(1, 3):contiguous()
   self.cudaGradOutput = self.cudaGradOutput:transpose(3, 1)
   self.cudaGradInput = self.cudaLayer:backward(self.cudaInput,
                                                self.cudaGradOutput)
   self:compareResults()
end


TestUpdateGradInputInPlace = {}

function TestUpdateGradInputInPlace:setUp()
   -- use setup from updateOutput() in-place tests
   TestUpdateOutputInPlace.setUp(self)
   -- run forward pass cuda
   self.cudaOutput = self.cudaLayer:forward(self.cudaInput)
   -- create gradOutput tensor and run backward pass
   self.hostGradOutput = torch.Tensor(16, 10, 10):float():uniform(-1.0, 1.0)
   self.hostGradInput = self.hostLayer:backward(self.hostInput,
                                                self.hostGradOutput)
   self.cudaGradOutput = self.hostGradOutput:cuda()
end

function TestUpdateGradInputInPlace:tearDown()
   collectgarbage()
end

function TestUpdateGradInputInPlace:compareResults()
   TestUpdateGradInput.compareResults(self)
end

function TestUpdateGradInputInPlace:testBase()
   self.cudaGradInput = self.cudaLayer:backward(self.cudaInput,
                                                self.cudaGradOutput)
   self:compareResults()
end

function TestUpdateGradInputInPlace:testPreserveTransposition()
   self.cudaGradOutput = self.cudaGradOutput:transpose(1, 3):contiguous()
   self.cudaGradOutput = self.cudaGradOutput:transpose(3, 1)
   self.cudaGradInput = self.cudaLayer:backward(self.cudaInput,
                                                self.cudaGradOutput)
   assert(self.cudaGradOutput:dim() == self.cudaGradInput:dim())
   for i = 1, self.cudaGradOutput:dim() do
      assert(self.cudaGradOutput:size(i) == self.cudaGradInput:size(i))
      assert(self.cudaGradOutput:stride(i) == self.cudaGradInput:stride(i))
   end
end

TestInPlaceCPU = {}

function TestInPlaceCPU:setUp()
   self.threshold = nn.Threshold(2.0, 1.0)
   self.input = torch.Tensor(16, 10, 10):uniform(-1.0, 3.0)
   self.inputIP = self.input:clone()
   assertTensorEq(self.input, self.inputIP, 'inputs must match')
   self.thresholdIP = nn.Threshold(2.0, 1.0, true)

end

function TestUpdateOutput:tearDown()
   collectgarbage()
end

function TestInPlaceCPU:testForward()
   local output = self.threshold:forward(self.input)
   local outputIP = self.thresholdIP:forward(self.inputIP)

   assertTensorEq(outputIP, output, 'in-place output wrong')
   assertTensorEq(outputIP, self.inputIP,
                  'in-place input must be same as output')
end

function TestInPlaceCPU:testBackward()
   local output   = self.threshold:forward(self.input)
   local outputIP = self.thresholdIP:forward(self.inputIP)

   output:uniform(-1.0, 1.0)
   outputIP = output:clone()

   local gradInput = self.threshold:backward(self.input, output)
   local gradInputIP = self.thresholdIP:backward(self.inputIP, outputIP)

   assertTensorEq(gradInputIP, gradInput, 'in-place result wrong')
   assertTensorEq(gradInputIP, outputIP,
                  'in-place input and output must be identical')
end

-- -----------------------------------------------------------------------------
-- Performance
-- -----------------------------------------------------------------------------

local function format_time(time, size)
   return string.format('%1.5E / %u = %1.5E', time, size, time/size)
end

PerformanceThresholdGPU = {}

function PerformanceThresholdGPU:tearDown()
   collectgarbage()
end

local function thresholdPerf(size, offset)
   offset = offset or 0.0
   local input = torch.Tensor(size):uniform(-1.0 + offset, 1.0 + offset):cuda()
   local threshold = nn.Threshold(offset, offset):cuda()

   threshold:updateOutput(input)
   cutorch.synchronize()
   local timer = torch.Timer()
   threshold:updateOutput(input)
   cutorch.synchronize()
   print(format_time(timer:time().real, size))
end

function PerformanceThresholdGPU:testSize1000()
   thresholdPerf(1000)
end

function PerformanceThresholdGPU:testSize10000()
   thresholdPerf(10000)
end

function PerformanceThresholdGPU:testSize100000()
   thresholdPerf(100000)
end

function PerformanceThresholdGPU:testSize1000000()
   thresholdPerf(1000000)
end

function PerformanceThresholdGPU:testSize10000000()
   thresholdPerf(10000000)
end

function PerformanceThresholdGPU:testSize100000000()
   thresholdPerf(100000000)
end

function PerformanceThresholdGPU:testSize1000000000()
   thresholdPerf(1000000000)
end

LuaUnit:main()

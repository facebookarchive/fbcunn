-- Copyright 2004-present Facebook. All Rights Reserved.

local mk = require('multikey')

-- Hoist this in a global buffer module
cudaTensorBuffers = {}
FFTConvolution = 'FFTConvolutionBuffer'
FFTConvolutionTranspose = 'FFTConvolutionTransposeBuffer'
FFTInputBufferType = 0
FFTInputTransposeBufferType = 1
FFTOutputBufferType = 2
FFTOutputTransposeBufferType = 3
FFTWeightBufferType = 4
FFTWeightTransposeBufferType = 5

-- Float assumed, 4 bytes
sizeOfElem = 4

local SpatialConvolutionCuFFT, parent =
      torch.class('nn.SpatialConvolutionCuFFT', 'nn.Module')

local precision = 0.00002
local printDebug = false
local debug = false

function SpatialConvolutionCuFFT:__init(nInputPlane, nOutputPlane,
                                        kW, kH, dW, dH)
   parent.__init(self)

   self.nInputPlane = nInputPlane
   self.nOutputPlane = nOutputPlane
   self.kW = kW
   self.kH = kH
   self.dW = dW or 1
   self.dH = dH or 1

   assert(self.dW == 1, "fft only supports stride-1 convolutions atm")

   self.weight = torch.Tensor(nOutputPlane, nInputPlane, kH, kW)
   self.bias = torch.Tensor(nOutputPlane)
   self.gradWeight = torch.Tensor(nOutputPlane, nInputPlane, kH, kW)
   self.gradBias = torch.Tensor(nOutputPlane)

   self:reset()
end

function SpatialConvolutionCuFFT:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1/math.sqrt(self.kW*self.kH*self.nInputPlane)
   end
   if nn.oldSeed then
      self.weight:apply(function()
         return torch.uniform(-stdv, stdv)
      end)
      self.bias:apply(function()
         return torch.uniform(-stdv, stdv)
      end)
   else
      self.weight:uniform(-stdv, stdv)
      self.bias:uniform(-stdv, stdv)
   end
end

local function debugVSMM(pass, module, toTest, fun, param1, param2, param3)
   local o = toTest:float():clone()
   toTest:zero()
   module.padding = 0
   module.finput = torch.CudaTensor()
   module.fgradInput = torch.CudaTensor()
   -- linearize weight for MM
   module.gradWeight =
      module.gradWeight:view(module.nOutputPlane,
                             module.nInputPlane * module.kH * module.kW)
   module.weight =
      module.weight:view(module.nOutputPlane,
                         module.nInputPlane * module.kH * module.kW)
   local test = fun(module, param1, param2, param3)
   -- reset layout of weight after MM
   module.gradWeight =
      module.gradWeight:view(module.nOutputPlane,
                             module.nInputPlane,
                             module.kH,
                             module.kW)
   module.weight =
      module.weight:view(module.nOutputPlane,
                         module.nInputPlane,
                         module.kH,
                         module.kW)
   local norm = math.sqrt(test:float():dot(test:float()) + 1e-8)
   if test:float():dist(o:float()) / norm > precision then
      print('error ', pass, test:float():dist(o:float()) / norm, precision)
      os.exit()
   elseif printDebug then
      print('debug vs MM check passes ',
            pass, o:min(), o:max(), o:mean(), o:std(), o:sum())
   end
end

function SpatialConvolutionCuFFT:updateOutput(input)
   self:prepareBuffers(input:size())
   input.nn.SpatialConvolutionCuFFT_updateOutput(self, input)

   if debug == true then
      debugVSMM("updateOutput",
                self,
                self.output,
                input.nn.SpatialConvolutionMM_updateOutput,
                input)
   end

   return self.output
end

function SpatialConvolutionCuFFT:explorePerformance(input, batches,
      inputs, planes, inputRows, inputCols, kernelRows, kernelCols)
   input.nn.SpatialConvolutionCuFFT_explorePerformance(self, batches,
      inputs, planes, inputRows, inputCols, kernelRows, kernelCols)
end

function SpatialConvolutionCuFFT:cleanupBuffers(input)
   input.nn.SpatialConvolutionCuFFT_cleanupBuffers()
end

function SpatialConvolutionCuFFT:updateGradInput(input, gradOutput)
   self:prepareBuffers(input:size())
   input.nn.SpatialConvolutionCuFFT_updateGradInput(self, gradOutput)

   if debug == true then
      debugVSMM("updateGradInput",
                self,
                self.gradInput,
                input.nn.SpatialConvolutionMM_updateGradInput,
                input,
                gradOutput)
   end

   return self.gradInput
end

local
function wrapMM_accGradParameters_gradWeight(module, input, gradOutput, scale)
   input.nn.SpatialConvolutionMM_accGradParameters(
      module, input, gradOutput, scale)
   return module.gradWeight
end

local
function wrapMM_accGradParameters_gradBias(module, input, gradOutput, scale)
   input.nn.SpatialConvolutionMM_accGradParameters(
      module, input, gradOutput, scale)
   return module.gradBias
end

function SpatialConvolutionCuFFT:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   self:prepareBuffers(input:size())
   input.nn.SpatialConvolutionCuFFT_accGradParameters(
     self, input, gradOutput, scale)

   if debug == true then
      self.gradBias:zero() -- zero first to avoid accumulation
      debugVSMM("accGradParameters_gradWeight",
                self,
                self.gradWeight,
                wrapMM_accGradParameters_gradWeight,
                input,
                gradOutput,
                scale)
      local saveBias = self.gradBias:float():clone()
      self.gradWeight:zero()
      self.gradBias:zero()
      debugVSMM("accGradParameters_gradBias",
                self,
                saveBias,
                wrapMM_accGradParameters_gradBias,
                input,
                gradOutput,
                scale)
   end
end

-- Type: input/gradInput, output/gradOutput or weight/gradWeight
-- Could lookup bit operations in lua and do in 1 line, just use a loop atm
local function nextPowerOf2(val)
   for i = 1, 10 do
      if (2 ^ i) >= val then
         return (2 ^ i)
      end
   end
   assert(false, 'Too large a convolution dimensions: ', val)
end


function SpatialConvolutionCuFFT:prepareBuffers(inputSize)
   self.inputBuffer = getBuffer(FFTConvolution, FFTInputBufferType, inputSize)
   self.inputTransposeBuffer = getBuffer(
      FFTConvolutionTranspose, FFTInputTransposeBufferType, inputSize)

   bufferSizesO = torch.LongStorage(4)
   bufferSizesO[1] = inputSize[1]     -- batch
   bufferSizesO[2] = self.nOutputPlane -- output planes
   bufferSizesO[3] = inputSize[3]     -- input x is always max for buffer
   bufferSizesO[4] = inputSize[4]     -- input y is always max for buffer
   self.outputBuffer = getBuffer(FFTConvolution, FFTOutputBufferType, bufferSizesO)
   self.outputTransposeBuffer = getBuffer(
      FFTConvolutionTranspose, FFTOutputTransposeBufferType, bufferSizesO)

   bufferSizesW = torch.LongStorage(4)
   bufferSizesW[1] = self.nOutputPlane -- output planes
   bufferSizesW[2] = self.nInputPlane  -- input planes
   bufferSizesW[3] = inputSize[3]     -- input x is always max for buffer
   bufferSizesW[4] = inputSize[4]     -- input y is always max for buffer
   self.weightBuffer = getBuffer(FFTConvolution, FFTWeightBufferType, bufferSizesW)
   self.weightTransposeBuffer = getBuffer(
      FFTConvolutionTranspose, FFTWeightTransposeBufferType, bufferSizesW)

   if self.inputBuffer and self.inputTransposeBuffer and
      self.outputBuffer and self.outputTransposeBuffer and
      self.weightBuffer and self.weightTransposeBuffer then
         return true
   end

   -- From here on, we should find failsafe to another SpatialConvolution
   self.inputBuffer = nil
   self.inputTransposeBuffer = nil
   self.outputBuffer = nil
   self.outputTransposeBuffer = nil
   self.weightBuffer = nil
   self.weightTransposeBuffer = nil
   freeBuffer(FFTConvolution, FFTInputBufferType, inputSize)
   freeBuffer(FFTConvolutionTranspose, FFTInputTransposeBufferType, inputSize)
   freeBuffer(FFTConvolution, FFTOutputBufferType, bufferSizesO)
   freeBuffer(FFTConvolutionTranspose, FFTOutputTransposeBufferType, bufferSizesO)
   freeBuffer(FFTConvolution, FFTWeightBufferType, bufferSizesW)
   freeBuffer(FFTConvolutionTranspose, FFTWeightTransposeBufferType, bufferSizesW)

   collectgarbage()
   collectgarbage()

   return false
end

function getBuffer(OperationType, tensorType, tensorSizes)
   d1 = tensorSizes[1]
   d2 = tensorSizes[2]
   -- Preemptively resize to d1 . d2 . 2^x . 2^y
   d3 = math.max(nextPowerOf2(tensorSizes[3]), nextPowerOf2(tensorSizes[4]))
   d4 = d3
   assert(d3 == d4, 'Squared fft convolution to support fbfft')
   numElements = d1 * d2 * d3 * (d4 / 2 + 1) * 2

   storage = torch.LongStorage(5)

   storage[1] = d1
   storage[2] = d2
   storage[3] = d3
   storage[4] = d4 / 2 + 1
   storage[5] = 2

   -- Conservative max buffer size, always needed at least by fbfft
   -- Handle memory bloat by tiled convolutions + inplace fft
   if mk.get(cudaTensorBuffers,
             OperationType,
             tensorType,
             cutorch.getDevice()) == nil then
      local free_bytes, total_bytes = cutorch.getMemoryUsage()
      if numElements * sizeOfElem > free_bytes then
         return nil
      end

      mk.put(cudaTensorBuffers, OperationType, tensorType, cutorch.getDevice(),
             torch.CudaTensor(storage))
   else
      -- Storage already exists but may need resizing.
      -- If resizing means expanding, make sure we have enough space
      t = mk.get(cudaTensorBuffers, OperationType, tensorType, cutorch.getDevice())
      if numElements > t:nElement() then
         -- Don't call cuda API unless really needed
         local free_bytes, total_bytes = cutorch.getMemoryUsage()
         if (numElements - t:nElement()) * sizeOfElem > free_bytes then
            return nil
         end
      end
      t:resize(storage)
   end

   t = mk.get(cudaTensorBuffers, OperationType, tensorType, cutorch.getDevice())
   return t
end

function freeBuffer(OperationType, tensorType, tensorSizes)
   mk.put(cudaTensorBuffers,
          OperationType,
          tensorType,
          cutorch.getDevice(), nil)
end

-- Copyright 2004-present Facebook. All Rights Reserved.

-- TODO: Catch errors in general
-- TODO: Catch errors on cufft plan creation and cleanupBuffers
-- TODO: Cleanup buffers and make them independent of tasks
-- TODO: Auto-tuning

require 'cudnn'
local List = require 'pl.List'
local thrift = require('fb.thrift')

-- Float assumed, 4 bytes
local sizeOfElem = 4

local prec = 0.00002

local function isnan(n) return tostring(n) == tostring((-1)^.5) end

-- Module

local SpatialConvolutionFFT, parent =
   torch.class('nn.SpatialConvolutionFFT', 'nn.Module')

-- multi-key map indexed by {BufferType, deviceId, [size], [metaData]}
SpatialConvolutionFFT.cudaTensorBuffers = {}
SpatialConvolutionFFT.bufferMap = {}

-- BufferType
SpatialConvolutionFFT.FFTInputBufferType =
   "FFTInputBufferType"
SpatialConvolutionFFT.FFTOutputBufferType =
   "FFTOutputBufferType"
SpatialConvolutionFFT.FFTWeightBufferType =
   "FFTWeightBufferType"
SpatialConvolutionFFT.CuFFTInputTransposeBufferType =
   "CuFFTInputTransposeBufferType"
SpatialConvolutionFFT.CuFFTOutputTransposeBufferType =
   "CuFFTOutputTransposeBufferType"
SpatialConvolutionFFT.CuFFTWeightTransposeBufferType =
   "CuFFTWeightTransposeBufferType"
SpatialConvolutionFFT.CuFFTPaddedInputBuffer =
   "CuFFTPaddedInputBuffer"
SpatialConvolutionFFT.CuFFTPaddedWeightBuffer =
   "CuFFTPaddedWeightBuffer"
SpatialConvolutionFFT.CuFFTPaddedOutputBuffer =
   "CuFFTPaddedOutputBuffer"

-- Convenience lists
SpatialConvolutionFFT.cudaRealBufferTypes = List{
   SpatialConvolutionFFT.CuFFTPaddedInputBuffer,
   SpatialConvolutionFFT.CuFFTPaddedWeightBuffer,
   SpatialConvolutionFFT.CuFFTPaddedOutputBuffer}
SpatialConvolutionFFT.cudaPaddedBufferTypes = List{
   SpatialConvolutionFFT.CuFFTPaddedInputBuffer,
   SpatialConvolutionFFT.CuFFTPaddedWeightBuffer,
   SpatialConvolutionFFT.CuFFTPaddedOutputBuffer}

-- Memory reuse policy
SpatialConvolutionFFT.memoryReuseNone = "none"
SpatialConvolutionFFT.memoryReuseInput = "input"
SpatialConvolutionFFT.memoryReuseOutput = "output"
SpatialConvolutionFFT.memoryReuseWeight = "weight"
SpatialConvolutionFFT.memoryReuseAll = "all"

-- Use to uniquely identify steps of this module and to properly track
-- producer-consumer dependences in the tagspace.
-- TODO: increment atomically in a multi-threaded environment
SpatialConvolutionFFT.moduleInstance = 0

-- Debug helper functions
local function wrapCUDNN_accGradParameters_gradWeight(
      module, input, gradOutput, scale)
   -- Needed to initialize all cudnn state properly
   module:updateOutput(input)
   module.gradBias:zero()
   module.gradWeight:zero()
   module:accGradParameters(input, gradOutput, scale)
   return module.gradWeight
end

local function wrapCUDNN_accGradParameters_gradBias(
      module, input, gradOutput, scale)
   -- Needed to initialize all cudnn state properly
   module:updateOutput(input)
   module.gradBias:zero()
   module.gradWeight:zero()
   module:accGradParameters(input, gradOutput, scale)
   return module.gradBias
end

function SpatialConvolutionFFT:debugVSCUDNN(
      pass, module, selfModule, toTest, fun, param1, param2, param3)
   local fftRes = toTest:float():clone()

   module.weight = selfModule.weight:clone()
   module.bias = selfModule.bias:clone()
   module.gradWeight = selfModule.gradWeight:clone()
   module.gradBias = selfModule.gradBias:clone()
   module.output = selfModule.output:clone()
   module.gradInput = selfModule.gradInput:clone()

   local p1 = param1:contiguous()
   local p2
   if param2 then
      p2 = param2:contiguous()
   end
   local p3 = param3
   local cudnnRes = fun(module, p1, p2, p3)

   if self.printDebugLevel >= 2 then
      print('FFTRES', {fftRes}, 'CUDNN', {cudnnRes})
   end

   local norm = math.sqrt(cudnnRes:float():dot(cudnnRes:float()) + 1e-8)
   if isnan(fftRes:sum()) or
   cudnnRes:float():dist(fftRes:float()) / norm > prec then
      print(torch.type(self), ' error', pass,
            cudnnRes:float():dist(fftRes:float()) / norm, prec)
      print(torch.type(self), ' error', pass,
            fftRes:min(), fftRes:max(), fftRes:mean(), fftRes:sum())
      if self.printDebugLevel >= 2 then
         local diff = fftRes:float() - cudnnRes:float()
         print('Expected\n', cudnnRes:float())
         print('Actual\n', fftRes:float())
         print('DIFFTENSOR\n', diff)
      end
      return false
   elseif self.printDebugLevel >= 0 then
      print(torch.type(self), ' debug vs CUDNN check passes ',
            pass, fftRes:min(), fftRes:max(), fftRes:mean(), fftRes:sum())
   end
   return true
end

function SpatialConvolutionFFT:initCudaResources(numHandles, numStreams)
   -- Init streams, handles and synchronization groups
   cutorch.reserveBlasHandles(numHandles)
   cutorch.reserveStreams(numStreams)
   local allStreams = {}
   for stream = 0, numStreams do
      table.insert(allStreams, stream)
   end
   local allStreamsButDefault = {}
   for stream = 1, numStreams do
      table.insert(allStreamsButDefault, stream)
   end
   return allStreams, allStreamsButDefault
end

function SpatialConvolutionFFT:__init(nInputPlane,
                                      nOutputPlane,
                                      kW,
                                      kH,
                                      dW,
                                      dH,
                                      padLeft,
                                      padUp,
                                      memoryReusePolicy,
                                      numCudaStreams)
   parent.__init(self)

   self.printDebugLevel = -1 -- override manually
   self.cudnnDebug = false -- override manually
   self.nInputPlane = nInputPlane
   self.nOutputPlane = nOutputPlane
   self.kW = kW
   self.kH = kH
   self.dW = dW or 1
   self.dH = dH or 1

   self.padLeft = padLeft or 0
   self.padUp = padUp or 0
   self.padRight = self.padLeft
   self.padDown = self.padUp

   assert(self.dW == 1, "fft only supports stride-1 convolutions atm")

   self.weight = torch.Tensor(nOutputPlane, nInputPlane, kH, kW)
   self.bias = torch.Tensor(nOutputPlane)
   self.gradWeight = torch.Tensor(nOutputPlane, nInputPlane, kH, kW)
   self.gradBias = torch.Tensor(nOutputPlane)

   -- Temporary buffers, would be nice to reduce code size here
   self.inputBuffer  = nil
   self.inputTransposeBuffer  = nil
   self.inputPadded  = nil
   self.outputBuffer = nil
   self.outputTransposeBuffer = nil
   self.outputPadded = nil
   self.weightBuffer = nil
   self.weightTransposeBuffer = nil
   self.weightPadded = nil

   -- CuFFT plans, useless for fbfft
   self.cufftPlanInputFFT = nil
   self.cufftPlanWeightFFT = nil
   self.cufftPlanOutputFFT = nil
   self.cufftPlanInputIFFT = nil
   self.cufftPlanWeightIFFT = nil
   self.cufftPlanOutputIFFT = nil

   self:reset()

   self.numCudaStreams = numCudaStreams or 16
   self.numCublasHandles = self.numCudaStreams
   self.allStreams = nil
   self.allStreamsButDefault = nil
   self.allStreams, self.allStreamsButDefault =
      self:initCudaResources(self.numCublasHandles, self.numCudaStreams)

   -- List of buffers into multikey that we need to free
   self.bufferKeys = List{}

   -- Memory reuse strategy
   if not memoryReusePolicy or
         memoryReusePolicy == nn.SpatialConvolutionFFT.memoryReuseNone
   then
      self.memoryReusePolicy = List{nn.SpatialConvolutionFFT.memoryReuseNone}
   elseif memoryReusePolicy == nn.SpatialConvolutionFFT.memoryReuseAll
   then
      self.memoryReusePolicy = List{nn.SpatialConvolutionFFT.memoryReuseInput,
                                    nn.SpatialConvolutionFFT.memoryReuseOutput,
                                    nn.SpatialConvolutionFFT.memoryReuseWeight}
   elseif torch.type(self.memoryReusePolicy) == 'table'
   then
      if memoryReusePolicy:contains(nn.SpatialConvolutionFFT.memoryReuseAll)
      then
         self.memoryReusePolicy =
            List{nn.SpatialConvolutionFFT.memoryReuseInput,
                 nn.SpatialConvolutionFFT.memoryReuseOutput,
                 nn.SpatialConvolutionFFT.memoryReuseWeight}
      else
         self.memoryReusePolicy = memoryReusePolicy
      end
   else
      self.memoryReusePolicy = List{memoryReusePolicy}
   end

   -- Use to uniquely identify steps of this module and to properly track
   -- producer-consumer dependences in the tagspace.
   SpatialConvolutionFFT.moduleInstance =
      SpatialConvolutionFFT.moduleInstance + 1 -- TODO: increment atomically
   -- Must be a unique name
   self.moduleUID =
      torch.type(self) .. "--instance=" .. SpatialConvolutionFFT.moduleInstance
   -- set once at the beginning of every operation to keep track of the
   -- 'timestep'
   self.timeSteps =
      { updateOutput = 0, updateGradInput = 0, accGradParameters = 0 }

   if self.printDebugLevel >= 0 then
      print('Post init ', self.moduleUID, ' memory usage: ',
            cutorch.getMemoryUsage())
   end

   -- List of fallback modules, one for each function (updateOutput,
   -- updateGradInput, accGradParameters)
   -- When they are set, just use the specified fallback for each pass.
   self.fallbackModules = nil
   self.recoverFromError = true

   -- Check vs reference result
   self.cudnnChecks = true

   -- Support for tuned SpatialConvolution.lua
   self.success = true
   self.autotuningPass = false
   self.reportErrors = true
end

function SpatialConvolutionFFT:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1/math.sqrt(self.kW * self.kH * self.nInputPlane)
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

-- Update output (i.e. forward prop)
function SpatialConvolutionFFT:updateOutput(input)
   self.timeSteps.updateOutput = self.timeSteps.updateOutput + 1
   self.originalStream = cutorch.getStream()
   local res = self:wrapFallback(self.updateOutputFFT, input)
   cutorch.setStream(self.originalStream)
   return res
end

function SpatialConvolutionFFT:updateGradInput(input, gradOutput)
   self.timeSteps.updateGradInput = self.timeSteps.updateGradInput + 1
   self.originalStream = cutorch.getStream()
   local res = self:wrapFallback(self.updateGradInputFFT, input, gradOutput)
   cutorch.setStream(self.originalStream)
   return res
end

function SpatialConvolutionFFT:accGradParameters(
      input, gradOutput, scale)
   self.timeSteps.accGradParameters = self.timeSteps.accGradParameters + 1
   self.originalStream = cutorch.getStream()
   self:wrapFallback(
      self.accGradParametersFFT, input, gradOutput, scale)
   cutorch.setStream(self.originalStream)
end

-- This function wraps calls to updateOutput, updateGradInput and
-- accGradParameters. If any error is encountered it cleans after itself and
-- calls the corresponding cudnn function. This acts as a failsafe mechanism in
-- case FFT runs out of memory which is not a trivial thing to determine
-- beforehand. The overhead is only paid on the first invocations, all
-- subsequent ones will default to cudnn after the first failure.
function SpatialConvolutionFFT:wrapFallback(
      fun, input, gradOutput, scale, reuseList)

   if not self.fallbackModules then
      local ok, res = pcall(fun, self, input, gradOutput, scale, reuseList)
      if ok then
         return res
      end
      if not self.recoverFromError then
         error(res)
      end

      if self.reportErrors then
         print("Error: " .. res .. " -> fallback to cudnn")
      end
      -- This path exits early for tuned SpatialConvolution.lua
      self.success = false
      if self.autotuningPass then
         if self.reportErrors then
            print('Using tuned SpatialConvolution: found an error, early exit')
         end
         return nil
      end
   end

   -- This path is the fallback path where cudnn is subsituted for our module
   -- This is becoming obsolete as everyone should now use
   -- tuned SpatialConvolution.lua
   if not self.collectedGarbage then
      self:cleanupBuffers()
      collectgarbage()
      collectgarbage()
      self.collectedGarbage = true
   end

   self.fallbackModules = {}
   if not self.fallbackModules[fun] then
      cutorch.synchronize()
      self.fallbackModules[fun] = cudnn.SpatialConvolution(self.nInputPlane,
                                                     self.nOutputPlane,
                                                     self.kW,
                                                     self.kH,
                                                     self.dW,
                                                     self.dH,
                                                     self.padLeft,
                                                     self.padUp):cuda()
      -- run updateOutput once to initialize
      self.fallbackModules[fun]:updateOutput(input)
   end

   -- Pass along to cudnn module
   self.fallbackModules[fun].weight = self.weight
   self.fallbackModules[fun].bias = self.bias
   self.fallbackModules[fun].gradWeight = self.gradWeight
   self.fallbackModules[fun].gradBias = self.gradBias
   local res = nil
   if fun == self.updateOutputFFT then
      res = self.fallbackModules[fun]:updateOutput(input)
      self.output = res
   elseif fun == self.updateGradInputFFT then
      res = self.fallbackModules[fun]:updateGradInput(input, gradOutput)
      self.gradInput = res
   elseif fun == self.accGradParametersFFT then
      self.fallbackModules[fun]:accGradParameters(input, gradOutput, scale)
      self.gradWeight = self.fallbackModules[fun].gradWeight
      self.gradBias = self.fallbackModules[fun].gradBias
   else
      error('Unknown call ' .. fun)
   end
   return res
end

function SpatialConvolutionFFT:getNormalizationFactor(commonSizes, input)
   if self.fftImplementation == 'fbfft' then
      return commonSizes[3] * commonSizes[4]
   elseif self.fftImplementation then
      return (input:size(3) + self.padUp + self.padDown) *
         (input:size(4) + self.padLeft + self.padRight)
   end
   error("Unknown fftImpl: " .. self.fftImplementation)
end

function SpatialConvolutionFFT:backward(input, gradOutput, scale)
   self.originalStream = cutorch.getStream()
   scale = scale or 1
   self:updateGradInput(input, gradOutput)
   self:wrapFallback(self.accGradParametersFFT,
                     input,
                     gradOutput,
                     scale,
                     List{self.outputTransposeBuffer})
   cutorch.setStream(self.originalStream)
   return self.gradInput
end

function SpatialConvolutionFFT:updateOutputFFTImpl()
   assert(false, 'This is an abstract class, must use a derived implementation')
end

function SpatialConvolutionFFT:updateGradInputFFTImpl()
   assert(false, 'This is an abstract class, must use a derived implementation')
end

function SpatialConvolutionFFT:accGradParametersFFTImpl()
   assert(false, 'This is an abstract class, must use a derived implementation')
end

function SpatialConvolutionFFT:updateOutputFFT(input, reuseList)
   assert(torch.type(input) == 'torch.CudaTensor', "CUDA support only!")
   local nBatches = input:size(1)

   -- Allocate the output for this module, only once
   if not self.output or self.output:nElement() == 0 then
      self.output = torch.CudaTensor(torch.LongStorage({
         nBatches,
         self.nOutputPlane,
         input:size(3) + self.padUp + self.padDown - self.kH + 1,
         input:size(4) + self.padLeft + self.padRight - self.kW + 1}))
   end

   if self.printDebugLevel >= 2 then
      print('PAD ', self.padUp, 'x', self.padLeft)
      print('ORIGINAL INPUT', {input})
      print('ORIGINAL WEIGHT', {self.weight})
      self.output:zero()
      print('ORIGINAL OUTPUT', {self.output})
   end

   -- Call the proper Impl
   self:updateOutputFFTImpl(input, reuseList)

   if self.printDebugLevel >= 0 then
      print('Post updateOutput ', self.moduleUID, ' memory usage: ',
            cutorch.getMemoryUsage())
   end

   if self.printDebugLevel >= 2 then
      print('FINAL INPUT', {input})
      print('COMPLEX INPUT POST FFT', {self.inputBuffer})
      print('COMPLEX INPUT POST TRANSPOSE', {self.inputTransposeBuffer})
      print('ORIGINAL WEIGHT', {self.weight})
      print('COMPLEX WEIGHT POST FFT', {self.weightBuffer})
      print('COMPLEX WEIGHT POST TRANSPOSE', {self.weightTransposeBuffer})
      print('OUTPUT CPLX TRANSPOSE POST MM', {self.outputTransposeBuffer})
      print('OUTPUT COMPLEX POST TRANSPOSE', {self.outputBuffer})
      print('OUTPUT REAL', {self.output})
   end

   if self.cudnnDebug then
      local sp = cudnn.SpatialConvolution(self.nInputPlane,
                                          self.nOutputPlane,
                                          self.kW,
                                          self.kH,
                                          self.dW,
                                          self.dH,
                                          self.padLeft,
                                          self.padUp):cuda()
      self.cudnnChecks = self.cudnnChecks and
         self:debugVSCUDNN("updateOutput",
                           sp,
                           self,
                           self.output,
                           sp.updateOutput,
                           input)
      sp = nil
      collectgarbage()
      collectgarbage()
   end

   return self.output
end


-- Update input gradients
function SpatialConvolutionFFT:updateGradInputFFT(input, gradOutput, reuseList)
   assert(torch.type(input) == 'torch.CudaTensor', "CUDA support only!")
   local nBatches = input:size(1)
   -- Allocate the gradInput for this module, only once
   if not self.gradInput or self.gradInput:nElement() == 0 then
      self.gradInput = torch.CudaTensor(torch.LongStorage({
                                              nBatches,
                                              self.nInputPlane,
                                              input:size(3),
                                              input:size(4)}))
   end

   if self.printDebugLevel >= 2 then
      print('PAD ', self.padUp, 'x', self.padLeft)
      print('ORIGINAL gradOutput', gradOutput)
      print('ORIGINAL WEIGHT', self.weight)
      print('ORIGINAL GRADINPUT', self.gradInput)
   end

   -- Call the proper Impl
   self:updateGradInputFFTImpl(input, gradOutput, reuseList)

   if self.printDebugLevel >= 0 then
      print('Post updateGradInput ', self.moduleUID, ' memory usage: ',
            cutorch.getMemoryUsage())
   end

   if self.printDebugLevel >= 2 then
      print('COMPLEX WEIGHT POST FFT', self.weightBuffer)
      print('COMPLEX WEIGHT POST TRANSPOSE', self.weightTransposeBuffer)
      print('COMPLEX GRADOUTPUT POST FFT', self.outputBuffer)
      print('COMPLEX GRADOUTPUT POST TRANSPOSE', self.outputTransposeBuffer)
      print('GRADINPUT COMPLEX POST MM', self.inputTransposeBuffer)
      print('GRADINPUT COMPLEX PRE IFFT', self.inputBuffer)
      print('REAL GRADINPUT', self.gradInput)
      print('REAL GRADINPUT PADDED (cufft only)', self.inputPadded)
   end

   if self.cudnnDebug then
      local sp = cudnn.SpatialConvolution(self.nInputPlane,
                                          self.nOutputPlane,
                                          self.kW,
                                          self.kH,
                                          self.dW,
                                          self.dH,
                                          self.padLeft,
                                          self.padUp):cuda()
      self.cudnnChecks = self.cudnnChecks and
         self:debugVSCUDNN("updateGradInput",
                           sp,
                           self,
                           self.gradInput,
                           sp.updateGradInput,
                           input,
                           gradOutput)
      sp = nil
      collectgarbage()
      collectgarbage()
   end

   return self.gradInput
end


-- Accumulate weight gradients
function SpatialConvolutionFFT:accGradParametersFFT(
      input, gradOutput, scale, reuseList)
   assert(torch.type(input) == 'torch.CudaTensor', "CUDA support only!")
   if not self.gradWeight or self.gradWeight:nElement() == 0 then
      assert(false, "GradWeight must already be allocated at module creation")
   end

   if self.printDebugLevel >= 2 then
      print('PAD ', self.padUp, 'x', self.padLeft)
      print('ORIGINAL INPUT', {input})
      print('ORIGINAL OUTPUT', {gradOutput})
      print('ORIGINAL WEIGHT', {self.gradWeight})
   end

   -- Call the proper Impl
   self:accGradParametersFFTImpl(input, gradOutput, scale, reuseList)

   if self.printDebugLevel >= 0 then
      print('Post accGradParameters ', self.moduleUID, ' memory usage: ',
            cutorch.getMemoryUsage())
   end

   if self.printDebugLevel >= 2 then
      print('OUTPUT COMPLEX POST TRANSPOSE', {self.outputBuffer})
      print('OUTPUT CPLX TRANSPOSE POST MM', {self.outputTransposeBuffer})
      print('COMPLEX INPUT POST TRANSPOSE', {self.inputTransposeBuffer})
      print('COMPLEX INPUT POST FFT', {self.inputBuffer})
      print('COMPLEX WEIGHT POST FFT', {self.weightBuffer})
      print('COMPLEX WEIGHT POST TRANSPOSE', {self.weightTransposeBuffer})
      print('REAL GRADWEIGHT', {self.weightPadded})
      print('REAL GRADWEIGHT', {self.gradWeight})
      print("SCALE: " .. scale)
   end

   if self.cudnnDebug then
      local saveBias = self.gradBias:float():clone()
      local sp = cudnn.SpatialConvolution(self.nInputPlane,
                                          self.nOutputPlane,
                                          self.kW,
                                          self.kH,
                                          self.dW,
                                          self.dH,
                                          self.padLeft,
                                          self.padUp):cuda()
      self.cudnnChecks = self.cudnnChecks and
         self:debugVSCUDNN("accGradParameters_gradWeight",
                           sp,
                           self,
                           self.gradWeight,
                           wrapCUDNN_accGradParameters_gradWeight,
                           input,
                           gradOutput,
                           scale)

      self.cudnnChecks = self.cudnnChecks and
         self:debugVSCUDNN("accGradParameters_gradBias",
                           sp,
                           self,
                           saveBias,
                           wrapCUDNN_accGradParameters_gradBias,
                           input,
                           gradOutput,
                           scale)
      sp = nil
      collectgarbage()
      collectgarbage()
   end
end


-- Buffer creation and reuse given a size and a pass.
-- Different passes use different tensors as the 'output of the pass'.
--   SpatialConvolutionFFT.ForwardFFTPass -> output
--   SpatialConvolutionFFT.BackwardFFTPass -> input
--   SpatialConvolutionFFT.AccGradientFFTPass -> weight
-- The buffers corresponding to the tensors that is the 'output of the pass'
-- must be properly transposed in order for the CGemm call to be consistent.
-- This is a simple metadata transposition, might as well construct properly.
--
-- This function contains the least common denominator of buffers needed for
-- all implementations.

SpatialConvolutionFFT.ForwardFFTPass = 1
SpatialConvolutionFFT.BackwardFFTPass = 2
SpatialConvolutionFFT.AccGradientFFTPass = 3

-- Meta-data is user specific metadata which influences the lifetime of the
-- buffers. Atm this is SpatialConvolutionFFTTiled-specific but if the network
-- is not too large, especially with parallel containers, this is a good
-- opportunity to reuse FFT computations.
function SpatialConvolutionFFT:prepareBuffers(commonSize, pass, metaData)
   assert(commonSize and self.fftImplementation)
   assert(torch.type(metaData) == 'table', torch.type(metaData))

   local bufferSizesO = torch.LongStorage({
         commonSize[1], self.nOutputPlane, commonSize[3], commonSize[4]})
   local bufferSizesW = torch.LongStorage({
         self.nOutputPlane, self.nInputPlane, commonSize[3], commonSize[4]})

   self.inputBuffer =
      self:getBuffer(
         SpatialConvolutionFFT.FFTInputBufferType, -- buffer type
         commonSize,                               -- buffer size
         false,                                    -- transposeLayout
         metaData)                   -- SpatialConvolutionFFTTiled-specific
   self.outputBuffer =
      self:getBuffer(
         SpatialConvolutionFFT.FFTOutputBufferType,
         bufferSizesO,
         false,
         metaData)                   -- SpatialConvolutionFFTTiled-specific
   self.weightBuffer =
      self:getBuffer(
         SpatialConvolutionFFT.FFTWeightBufferType,
         bufferSizesW,
         false,
         metaData)                   -- SpatialConvolutionFFTTiled-specific

   if self.inputBuffer and self.outputBuffer and self.weightBuffer then
      return true
   end

   -- TODO: From here on, we should failsafe to another SpatialConvolution
   self:cleanupBuffers()

   error('Not enough memory for FFT buffers, need to fall back')
end


-- Returns nil if it cannot allocate a new buffer (for error recovery cases)
function SpatialConvolutionFFT:getBuffer(
      BufferType, tensorSizes, transposedLayout, metaData)
   assert(torch.type(metaData) == 'table', torch.type(metaData))

   local d1 = tensorSizes[1]
   local d2 = tensorSizes[2]
   local d3 = tensorSizes[3]
   local d4 = tensorSizes[4]

   local numElements = 0
   local sizes = torch.LongStorage({0})
   local isRealBuffer = SpatialConvolutionFFT.cudaRealBufferTypes:contains(
      BufferType)
   local isComplexBuffer = not isRealBuffer

   if isComplexBuffer then
      -- fbfft and cufft have different layouts
      assert(self.fftImplementation)
      if self.fftImplementation == 'fbfft' then
         numElements = d1 * d2 * (d3 / 2 + 1) * d4 * 2
         if transposedLayout then
            -- The buffers corresponding to the tensors that is the
            -- 'output of the pass' must be properly transposed in order for the
            -- CGemm call to be consistent.
            -- This is a simple metadata transposition, might as well construct
            -- properly.
            sizes = torch.LongStorage({d3 / 2 + 1, d4, d1, d2, 2})
         else
            sizes = torch.LongStorage({d1, d2, d3 / 2 + 1, d4, 2})
         end
      else
         numElements = d1 * d2 * d3 * (d4 / 2 + 1) * 2
         if transposedLayout then
            -- The buffers corresponding to the tensors that is the
            -- 'output of the pass' must be properly transposed in order for the
            -- CGemm call to be consistent.
            -- This is a simple metadata transposition, might as well construct
            -- properly.
            sizes = torch.LongStorage({d3, d4 / 2 + 1, d1, d2, 2})
         else
            sizes = torch.LongStorage({d1, d2, d3, d4 / 2 + 1, 2})
         end
      end
   else
      -- Real buffers, for padding purposes in first approx
      if self.fftImplementation == 'cufft' and
      SpatialConvolutionFFT.cudaPaddedBufferTypes:contains(BufferType) then
         numElements = d1 * d2 * d3 * d4
         -- TODO: potentially wasteful if original tensor is already of
         -- tensorSizes. Could clean this up but requires knowing the original
         -- tensor as a model for which we pad.
         sizes = torch.LongStorage({d1, d2, d3, d4})
      end
      -- else allocate an empty tensor, nil is reserved for errors
   end

   assert(sizes and #sizes > 0)

   -- Conservative max buffer size, always needed at least by fbfft
   -- Handle memory bloat by tiled convolutions + inplace fft
   local bufferKey = self:getBufferKey(BufferType, sizes, metaData)
   if SpatialConvolutionFFT.bufferMap[bufferKey] == nil then
      local free_bytes = cutorch.getMemoryUsage()
      if numElements * sizeOfElem > free_bytes then
         return nil
      end

      local before = cutorch.getMemoryUsage()
      SpatialConvolutionFFT.bufferMap[bufferKey] = torch.CudaTensor(sizes)
      local after = cutorch.getMemoryUsage()
      if self.printDebugLevel >= 1 then
         print('FFT Buffer Create Allocated ', before - after)
      end
   else
      -- Storage already exists but may need resizing.
      -- If resizing means expanding, make sure we have enough space
      local t = SpatialConvolutionFFT.bufferMap[bufferKey]
      if numElements > t:nElement() then
         -- Don't call cuda API unless really needed
         local free_bytes = cutorch.getMemoryUsage()
         -- Resize is not in place, need to hold both in memory at some point
         -- The subsequent resize cannot fail in cuda land or we're hosed and
         -- cudaGetLastError will be 2.
         if (numElements + t:nElement()) * sizeOfElem > free_bytes then
            assert(false, 'Out of memory: cannot hold both tensors for resize')
         end
         local before = cutorch.getMemoryUsage()
         t:resize(sizes)
         local after = cutorch.getMemoryUsage()
         if self.printDebugLevel >= 1 then
            print('FFT Buffer Resize Allocated ', before - after)
         end
      else
         -- Still need to resize to make the sizes / strides as expected but
         -- this does cost extra memory
         t:resize(sizes)
      end
   end

   local t = SpatialConvolutionFFT.bufferMap[bufferKey]
   assert(t, 'Tensor buffer improperly set')

   for d = 1, t:nDimension() do
      if (sizes[d] ~= t:size(d)) then
         print("Put / get buffer dimension mismatch! d = ", d, " expected = ",
               sizes, " actual = ", {t})
         assert(sizes[d] == t:size(d))
      end
   end

   return t
end

function SpatialConvolutionFFT:freeBuffer(bufferKey)
   local tensor = SpatialConvolutionFFT.bufferMap[bufferKey]
   if tensor then
      SpatialConvolutionFFT.bufferMap[bufferKey] = nil
   end
end

-- Returns a string key, not hashed atm.
-- For instance, in SpatialConvolutionFFTTiled, this helps the creation of
-- different buffers for various tile tensorSize, tileSize and tileIndices.
-- This is important in order to reuse frequency domain representation
-- of tiled pieces of the tensors.
-- This allows trading off reuse for memory consumption.
--
-- In FBFFT and CuFFT however, memory consumption can grow quickly so one should
-- only use a single buffer per BufferType.
-- If we had some user information that the buffers remain small enough, we
-- could have per module persistent buffers that would allow reuse.
function SpatialConvolutionFFT:getBufferKey(BufferType, bufferSizes, metaData)
   assert(false, "getBufferKey controls buffers lifetime: must be overridden")
end


-- This implementation reuses buffers and keeps memory consumption minimal
-- (but this can still be a lot).
-- In particular, we only discriminate buffers by deviceId and type of buffer
-- by default.
-- This means we only have 1 copy of each type of buffer per device.
-- The same buffers are reused across any call of any module so the only
-- possible reuse is the reuse of gradOutput in the backward function.
-- This requires that backward be properly implemented in container modules
-- to allow such reuse.
-- For more advanced reuses, a proper getBufferKey function needs to be
-- implemented, tradeoffs will be made between reuse and memory consumption.
function SpatialConvolutionFFT:getBufferKeyGeneric(BufferType)
   local bufferKey = {
      SpatialConvolutionFFT.cudaTensorBuffers,
      cutorch.getDevice(),
      BufferType,
   }
   local res = thrift.to_string(bufferKey)
   if not self.bufferKeys:contains(res) then
      self.bufferKeys:append(res)
   end
   return res
end

function SpatialConvolutionFFT:cleanupBuffers()
   -- release all local result tensors and all buffers
   self.output = nil
   self.gradInput = nil

   -- Kill local references to global buffers
   self.inputBuffer = nil
   self.outputBuffer = nil
   self.weightBuffer = nil

   -- Free all buffers
   local len = self.bufferKeys:len()
   for i = 1, len do
      self:freeBuffer(self.bufferKeys:pop())
   end

   self.fallbackModules = {}
   SpatialConvolutionFFT.cudaTensorBuffers = {}
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

function SpatialConvolutionFFT:prepareCuFFTSizeAndBuffers(
      i, w, o, metaData, pass)
   local commonSize = i:size()
   -- If we use cufft we should use rectangular regions where the width is a
   -- power of 2. This is usually good enough approximation between FFT
   -- efficiency and avoiding spurious work.
   commonSize[3] =
      math.max(i:size(3) + self.padUp + self.padDown,
               w:size(3),
               o:size(3))
   commonSize[4] =
      nextPowerOf2(math.max(i:size(4) + self.padLeft + self.padRight,
                            w:size(4),
                            o:size(4)))
   self:prepareBuffers(commonSize, pass, metaData)

   assert(self.fftImplementation == "cufft",
          "CuFFT convolution module expected!")
   assert(self.inputPadded and self.weightPadded and self.outputPadded,
          "CuFFT requires padded input, weight and output")

   if o == self.output then
      self.inputPadded:zero()
      self.weightPadded:zero()
   elseif w == self.weight then
      self.weightPadded:zero()
      self.outputPadded:zero()
   else
      self.inputPadded:zero()
      self.outputPadded:zero()
   end

   return commonSize -- needed for normalization factor
end

function SpatialConvolutionFFT:prepareFBFFTGemmSizeAndBuffers(
      i, w, o, metaData, pass)
   local commonSize = i:size()
   -- If we use cufft we should use rectangular regions where the width is a
   -- power of 2. This is usually good enough approximation between FFT
   -- efficiency and avoiding spurious work.
   commonSize[3] =
      nextPowerOf2(math.max(i:size(3) + self.padUp + self.padDown,
                            i:size(4) + self.padLeft + self.padRight,
                            w:size(3),
                            w:size(4),
                            o:size(3),
                            o:size(4)))
   commonSize[4] = commonSize[3]
   self:prepareBuffers(commonSize, pass, metaData)

   assert(self.fftImplementation == "fbfft",
          "FBFFT convolution module expected!")
   assert(not self.inputPadded and not self.weightPadded and
             not self.outputPadded,
          "CuFFT requires padded input, weight and output")

   return commonSize -- needed for normalization factor
end

local NO_TRANSPOSE = nil

-- Makes or reuses square FFT buffers up to the next power of 2
function SpatialConvolutionFFT:prepareFBFFTSizeAndBuffers(i, w, o, metaData)
   local commonSize = i:size()
   commonSize[3] =
      nextPowerOf2(math.max(i:size(3) + self.padUp + self.padDown,
                            i:size(4) + self.padLeft + self.padRight,
                            w:size(3),
                            w:size(4),
                            o:size(3),
                            o:size(4)))
   commonSize[4] = commonSize[3]
   self:prepareBuffers(commonSize, NO_TRANSPOSE, metaData)
   assert(self.fftImplementation == "fbfft",
          "FBFFT convolution module expected!")
   assert(not self.inputPadded and not self.weightPadded and
             not self.outputPadded,
          "FBFFT does not expect padded input, weight and output")
   return commonSize -- needed for normalization factor
end

function SpatialConvolutionFFT:setReuseInputs(val)
   assert(type(val) == 'boolean')
   self:_setReuse(val, nn.SpatialConvolutionFFT.memoryReuseInput)
end

function SpatialConvolutionFFT:setReuseOutputs(val)
   assert(type(val) == 'boolean')
   self:_setReuse(val, nn.SpatialConvolutionFFT.memoryReuseOutput)
end

function SpatialConvolutionFFT:setReuseWeights(val)
   assert(type(val) == 'boolean')
   self:_setReuse(val, nn.SpatialConvolutionFFT.memoryReuseWeight)
end

function SpatialConvolutionFFT:_setReuse(val, toReuse)
   assert(type(val) == 'boolean')
   assert(toReuse == nn.SpatialConvolutionFFT.memoryReuseInput or
             toReuse == nn.SpatialConvolutionFFT.memoryReuseOutput or
             toReuse == nn.SpatialConvolutionFFT.memoryReuseWeight,
          toReuse)

   if val then
      if self.memoryReusePolicy:contains(
            nn.SpatialConvolutionFFT.memoryReuseNone) then
         -- Override
         self.memoryReusePolicy = List{toReuse}
      elseif self.memoryReusePolicy:contains(toReuse) then
         -- Do nothing
         return
      else
         self.memoryReusePolicy:append(toReuse)
      end
   else
      if self.memoryReusePolicy:contains(toReuse) then
         self.memoryReusePolicy:remove_value(toReuse)
         -- Set at least "none"
         if self.memoryReusePolicy:len() == 0 then
            self.memoryReusePolicy:append(
               nn.SpatialConvolutionFFT.memoryReuseNone)
         end
      else
         -- Do nothing
         return
      end
   end
end

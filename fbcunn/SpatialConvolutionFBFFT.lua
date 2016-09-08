-- Copyright 2004-present Facebook. All Rights Reserved.

require 'cudnn'
local thrift = require('fb.thrift')
local ffi = require 'ffi'

local lib_name = 'torch_fb_fbcunn_mm'
local lib_path = package.searchpath(lib_name, package.cpath)
local FBMMFFI = ffi.load(lib_path and lib_path or lib_name)

local lib_name = 'torch_fb_fbcunn_convolution_bias'
local lib_path = package.searchpath(lib_name, package.cpath)
local ConvolutionBiasFFI = ffi.load(lib_path and lib_path or lib_name)

--[[
   Actual module
--]]
local SpatialConvolutionFBFFT, parent =
   torch.class('nn.SpatialConvolutionFBFFT', 'nn.SpatialConvolutionFFT')

-- memoryReusePolicy is one of:
--   SpatialConvolutionFFT.memoryReuseNone
--   SpatialConvolutionFFT.memoryReuseWeight
--   SpatialConvolutionFFT.memoryReuseInput
--   SpatialConvolutionFFT.memoryReuseOutput
function SpatialConvolutionFBFFT:__init(nInputPlane,
                                        nOutputPlane,
                                        kW,
                                        kH,
                                        dW,
                                        dH,
                                        padLeft,
                                        padUp,
                                        memoryReusePolicy,
                                        numCudaStreams)
   assert(torch.type(nInputPlane) == 'number')
   assert(torch.type(nOutputPlane) == 'number')
   assert(torch.type(kW) == 'number')
   assert(torch.type(kH) == 'number')
   assert(torch.type(dW) == 'number')
   assert(torch.type(dH) == 'number')
   assert(torch.type(padLeft) == 'number')
   assert(torch.type(padUp) == 'number')
   assert(memoryReusePolicy == nil or
             torch.type(memoryReusePolicy) == 'string' or
             torch.type(memoryReusePolicy) == 'table')
   assert(numCudaStreams == nil or torch.type(numCudaStreams) == 'number')

   parent.__init(
      self, nInputPlane, nOutputPlane, kW, kH, dW, dH, padLeft, padUp,
      memoryReusePolicy, numCudaStreams)
   parent.fftImplementation = 'fbfft'

   -- Sanity assertions
   assert(self.printDebugLevel == -1)
   assert(self.nInputPlane == nInputPlane)
   assert(self.nOutputPlane == nOutputPlane)
   assert(self.kW == kW)
   assert(self.kH == kH)
   assert(self.dH == 1, "fft only supports stride-1 convolutions atm")
   assert(self.dW == 1, "fft only supports stride-1 convolutions atm")

   assert(self.weight:size(1) == nOutputPlane and
             self.weight:size(2) == nInputPlane and
             self.weight:size(3) == kH and
             self.weight:size(4) == kW)
   assert(self.bias:size(1) == nOutputPlane)
   assert(self.gradWeight:size(1) == nOutputPlane and
             self.gradWeight:size(2) == nInputPlane and
             self.gradWeight:size(3) == kH and
             self.gradWeight:size(4) == kW)
   assert(self.gradBias:size(1) == nOutputPlane)

   -- Temporary buffers
   assert(not self.inputBuffer)
   assert(not self.inputTransposeBuffer)
   assert(not self.inputPadded)
   assert(not self.outputBuffer)
   assert(not self.outputTransposeBuffer)
   assert(not self.outputPadded)
   assert(not self.weightBuffer)
   assert(not self.weightTransposeBuffer)
   assert(not self.weightPadded)

   -- FBFFT plans, useless for fbfft
   assert(not self.cufftPlanInputFFT)
   assert(not self.cufftPlanWeightFFT)
   assert(not self.cufftPlanOutputFFT)
   assert(not self.cufftPlanInputIFFT)
   assert(not self.cufftPlanWeightIFFT)
   assert(not self.cufftPlanOutputIFFT)

   assert(self.padUp < self.kH and self.padDown < self.kH and
             self.padLeft < self.kW and self.padRight < self.kW,
          "Padding must be smaller than kernel")
end

function SpatialConvolutionFBFFT:prepareSizeAndBuffers(i, w, o, metaData)
   return self:prepareFBFFTSizeAndBuffers(i, w, o, metaData)
end

function SpatialConvolutionFBFFT:updateOutputFFTImpl(input, reuseList, metaData)
   assert(torch.type(input) == 'torch.CudaTensor', "CUDA support only!")

   if not metaData then
      metaData = {}
      metaData.pass = nn.SpatialConvolutionFFT.ForwardFFTPass
   end

   local commonSize =
      self:prepareSizeAndBuffers(input, self.weight, self.output, metaData)

   local cublasBatchDims = 2
   -- 2D convolutions on 4D tensors atm
   assert(#input:size() == cublasBatchDims + 2)

   local inputFFTStream = 1
   local weightFFTStream = 2
   local fbmmStream = 3
   assert(cutorch.getNumStreams() >= 3)

   -- Synchronize all streams on SESE, change when we have a proper DAG impl
   -- ##############################################
   cutorch.streamBarrier(self.allStreams)

   -- 1. FFTs
   if not reuseList or
      not reuseList:contains(nn.SpatialConvolutionFFT.FFTInputBufferType)
   then
      -- Potentially reuse buffer if so told
      -- Makes sense because we could asynchronously compute these AoT
      local fftWrapperPadded = nn.FFTWrapper(
         self.fftImplementation, self.padLeft, self.padUp)
      cutorch.setStream(inputFFTStream)
      fftWrapperPadded:fft(input, self.inputBuffer, cublasBatchDims)
      cutorch.setStream(fbmmStream)
      cutorch.streamWaitFor(fbmmStream, {inputFFTStream})
   end

   if not reuseList or
      not reuseList:contains(nn.SpatialConvolutionFFT.FFTWeightBufferType)
   then
      -- Potentially reuse buffer if so told
      -- Makes sense because we could asynchronously compute these AoT
      local fftWrapper = nn.FFTWrapper(self.fftImplementation)
      cutorch.setStream(weightFFTStream)
      fftWrapper:fft(self.weight, self.weightBuffer, cublasBatchDims)
      cutorch.setStream(fbmmStream)
      cutorch.streamWaitFor(fbmmStream, {weightFFTStream})
   end

   -- 2. GEMM with in place transpose
   -- stream must match the IFFT stream for sync without waiting
   -- explicitly
   cutorch.setStream(fbmmStream)
   local norm = self:getNormalizationFactor(commonSize, input)
   FBMMFFI.transposeMMFFI(cutorch._state,
                          self.inputBuffer:cdata(),
                          self.weightBuffer:cdata(),
                          self.outputBuffer:cdata(),
                          1.0 / norm,
                          false,
                          true,
                          false)

   -- 3. IFFT
   local fftWrapper = nn.FFTWrapper(self.fftImplementation)
   cutorch.setStream(fbmmStream)
   fftWrapper:ffti(self.output, self.outputBuffer, cublasBatchDims)

   -- 4. Finally, bias update
   if not metaData.skipBias then
      cutorch.setStream(fbmmStream)
      ConvolutionBiasFFI.updateOutputBiasFFI(
         cutorch._state, self.output:cdata(), self.bias:cdata())
   end

   -- Synchronize all streams on SESE, change when we have a proper DAG impl
   -- ##############################################
   cutorch.streamBarrier(self.allStreams)

   return self.output
end


--[[
   Update input gradients
--]]
function SpatialConvolutionFBFFT:updateGradInputFFTImpl(
      input, gradOutput, reuseList, metaData)
   assert(torch.type(input) == 'torch.CudaTensor', "CUDA support only!")

   if not metaData then
      metaData = {}
      metaData.pass = nn.SpatialConvolutionFFT.BackwardFFTPass
   end

   local commonSize =
      self:prepareSizeAndBuffers(input, self.weight, gradOutput, metaData)

   local cublasBatchDims = 2
   -- 2D convolutions on 4D tensors atm
   assert(#input:size() == cublasBatchDims + 2)

   local gradInputFFTStream = 1
   local gradOutputFFTStream = 2
   local fbmmStream = 3
   assert(cutorch.getNumStreams() >= 3)

   -- Synchronize all streams on SESE, change when we have a proper DAG impl
   -- ##############################################
   cutorch.streamBarrier(self.allStreams)

   -- 1. FFTs
   if (not reuseList or
          not reuseList:contains(nn.SpatialConvolutionFFT.FFTOutputBufferType))
   then
      -- Potentially reuse buffer if so told
      local fftWrapper = nn.FFTWrapper(self.fftImplementation)
      cutorch.setStream(gradOutputFFTStream)
      fftWrapper:fft(gradOutput, self.outputBuffer, cublasBatchDims)
      cutorch.setStream(fbmmStream)
      cutorch.streamWaitFor(fbmmStream, {gradOutputFFTStream})
   end

   if (not reuseList or
          not reuseList:contains(nn.SpatialConvolutionFFT.FFTWeightBufferType))
      and not self.memoryReusePolicy:contains(
         nn.SpatialConvolutionFFT.memoryReuseWeight)
   then
      -- Potentially reuse buffer if so told
      local fftWrapper = nn.FFTWrapper(self.fftImplementation)
      cutorch.setStream(gradInputFFTStream)
      fftWrapper:fft(self.weight, self.weightBuffer, cublasBatchDims)
      cutorch.setStream(fbmmStream)
      cutorch.streamWaitFor(fbmmStream, {gradInputFFTStream})
   end

   -- 2. GEMM with in place transpose
   -- stream must match the IFFT stream for sync without waiting
   -- explicitly
   cutorch.setStream(fbmmStream)
   local norm = self:getNormalizationFactor(commonSize, gradOutput)
   FBMMFFI.transposeMMFFI(cutorch._state,
                          self.outputBuffer:cdata(),
                          self.weightBuffer:cdata(),
                          self.inputBuffer:cdata(),
                          1.0 / norm,
                          false,
                          false,
                          false)

   -- 3. IFFT
   cutorch.setStream(fbmmStream)
   local fftWrapperPadded = nn.FFTWrapper(
         self.fftImplementation, self.padLeft, self.padUp)
   fftWrapperPadded:ffti(self.gradInput, self.inputBuffer, cublasBatchDims)

   -- ##############################################
   cutorch.streamBarrier(self.allStreams)

   return self.gradInput
end


--[[
   Accumulate weight gradients
--]]
function SpatialConvolutionFBFFT:accGradParametersFFTImpl(
      input, gradOutput, scale, reuseList, metaData)
   assert(torch.type(input) == 'torch.CudaTensor', "CUDA support only!")

   local scale = scale or 1

   if not metaData then
      metaData = {}
      metaData.pass = nn.SpatialConvolutionFFT.AccGradientFFTPass
   end

   local commonSize =
      self:prepareSizeAndBuffers(input, self.gradWeight, gradOutput, metaData)

   local cublasBatchDims = 2
   -- 2D convolutions on 4D tensors atm
   assert(#input:size() == cublasBatchDims + 2)

   local inputFFTStream = 1
   local gradOutputFFTStream = 2
   local gradBiasFFTStream = 3
   local fbmmStream = 4
   assert(cutorch.getNumStreams() >= 4)

   -- Synchronize all streams on SESE, change when we have a proper DAG impl
   -- #########################################
   cutorch.streamBarrier(self.allStreams)

   -- 0. Bias update is independent
   if not metaData.skipBias then
      cutorch.setStream(gradBiasFFTStream)
      ConvolutionBiasFFI.accGradParametersBiasFFI(
         cutorch._state, gradOutput:cdata(), self.gradBias:cdata(), scale)
   end

   -- 1. FFTs
   if (not reuseList or not reuseList:contains(
          nn.SpatialConvolutionFFT.FFTOutputBufferType)) and
      not self.memoryReusePolicy:contains(
         nn.SpatialConvolutionFFT.memoryReuseOutput)
   then
      -- Potentially reuse buffer if so told
      local fftWrapper = nn.FFTWrapper(self.fftImplementation)
      cutorch.setStream(gradOutputFFTStream)
      fftWrapper:fft(gradOutput, self.outputBuffer, cublasBatchDims)
      cutorch.setStream(fbmmStream)
      cutorch.streamWaitFor(fbmmStream, {gradOutputFFTStream})
   end

   if (not reuseList or not reuseList:contains(
          nn.SpatialConvolutionFFT.FFTInputBufferType)) and
      not self.memoryReusePolicy:contains(
         nn.SpatialConvolutionFFT.memoryReuseInput)
   then
      -- Potentially reuse buffer if so told
      cutorch.setStream(inputFFTStream)
      local fftWrapperPadded = nn.FFTWrapper(
         self.fftImplementation, self.padLeft, self.padUp)
      fftWrapperPadded:fft(input, self.inputBuffer, cublasBatchDims)
      cutorch.setStream(fbmmStream)
      cutorch.streamWaitFor(fbmmStream, {inputFFTStream})
   end

   -- 2. GEMM with in place transpose
   -- stream must match the IFFT stream for sync without waiting
   -- explicitly
   cutorch.setStream(fbmmStream)
   local norm = self:getNormalizationFactor(commonSize, gradOutput)
   FBMMFFI.transposeMMFFI(cutorch._state,
                          self.outputBuffer:cdata(),
                          self.inputBuffer:cdata(),
                          self.weightBuffer:cdata(),
                          (1.0 * scale) / norm,
                          true,
                          false,
                          false)

   -- 3. IFFT
   cutorch.setStream(fbmmStream)
   local fftWrapper = nn.FFTWrapper(self.fftImplementation)
   fftWrapper:ffti(self.gradWeight, self.weightBuffer, cublasBatchDims)

   -- Synchronize all streams on SESE, change when we have a proper DAG impl
   -- #########################################
   cutorch.streamBarrier(self.allStreams)
end


function SpatialConvolutionFBFFT:getBufferKey(BufferType, bufferSizes, metaData)
   assert(torch.type(bufferSizes) == 'torch.LongStorage',
          torch.type(bufferSizes))
   assert(torch.type(metaData) == 'table',
          torch.type(metaData))

   if self.memoryReusePolicy:contains(
      nn.SpatialConvolutionFFT.memoryReuseNone)
   then
      return parent.getBufferKeyGeneric(self, BufferType)
   end

   if not self.memoryReusePolicy:contains(
         nn.SpatialConvolutionFFT.memoryReuseWeight) and
      not self.memoryReusePolicy:contains(
         nn.SpatialConvolutionFFT.memoryReuseInput) and
      not self.memoryReusePolicy:contains(
         nn.SpatialConvolutionFFT.memoryReuseOutput)
   then
         assert(false, "unknown memory reuse policy " .. self.memoryReusePolicy)
   end

   -- TODO: needs semantics for proper producer consumer dependences and
   -- ordering for RAW dependences by using self.moduleTimeStep properly
   local md = {}
   if metaData then
      -- This is an adhoc way to discriminate between
      --   updateOutput   / updateGradInput      / accGradParameters
      --   input  (false) /   gradInput  (true)  / input      (false)
      --   output (true)  /   gradOutput (false) / input      (false)
      --   weight (false) /   weight     (false) / gradWeight (true)
      --
      local isOutputOfAlgorithm = false
      -- In cufft mode, the complex buffers are reused
      if (metaData.pass == nn.SpatialConvolutionFFT.ForwardFFTPass and
             BufferType == nn.SpatialConvolutionFFT.FFTOutputBufferType) or
         (metaData.pass == nn.SpatialConvolutionFFT.BackwardFFTPass and
             BufferType == nn.SpatialConvolutionFFT.FFTInputBufferType) or
         (metaData.pass == nn.SpatialConvolutionFFT.AccGradientFFTPass and
             BufferType == nn.SpatialConvolutionFFT.FFTWeightBufferType)
      then
         isOutputOfAlgorithm = true
      end
      md.isOutputOfAlgorithm = isOutputOfAlgorithm
   end

   -- If no memory reuse, all modules must use the same buffers, only
   -- discriminate by buffer type and device id.
   local moduleDiscr = self.moduleUID
   if self.memoryReusePolicy:contains(nn.SpatialConvolutionFFT.memoryReuseNone)
   then
      moduleDiscr = nil
      bufferSizes = nil
      md = nil
   end

   local bufferKey = {
      self.cudaTensorBuffers,
      cutorch.getDevice(),
      BufferType,
      bufferSizes,
      moduleDiscr,
      -- Be sure to put a counter for buffer and reuse btw timesteps or
      -- memory will be blown (i.e. full DSA = ouch)
      -- self.moduleTimeStep,
      md
   }
   local res = thrift.to_string(bufferKey)
   if not self.bufferKeys:contains(res) then
      self.bufferKeys:append(res)
   end
   return res
end

function SpatialConvolutionFBFFT:cleanupBuffers()
   parent.cleanupBuffers(self)
end

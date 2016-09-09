-- Copyright 2004-present Facebook. All Rights Reserved.

require 'cudnn'
local ffi = require 'ffi'
-- TODO: @soumith, any better way than this fully convoluted path ?
local ConvolutionBiasFFI = ffi.load('torch_fb_fbcunn_convolution_bias')
local thrift = require('fb.thrift')

ffi.cdef[[
   void updateOutputBiasFFI(THCState*, THCudaTensor*, THCudaTensor*);
   void accGradParametersBiasFFI(
      THCState*, THCudaTensor*, THCudaTensor*, float scale);
]]

--[[
   Actual module
--]]
local SpatialConvolutionFBFFTGemm, parent =
   torch.class('nn.SpatialConvolutionFBFFTGemm', 'nn.SpatialConvolutionFFT')

function SpatialConvolutionFBFFTGemm:__init(nInputPlane,
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
   assert(memoryReusePolicy == nil or
             torch.type(memoryReusePolicy) == 'string' or
             torch.type(memoryReusePolicy) == 'table')
   assert(numCudaStreams == nil or torch.type(numCudaStreams) == 'number')

   parent.__init(self,
                 nInputPlane,
                 nOutputPlane,
                 kW,
                 kH,
                 dW,
                 dH,
                 padLeft,
                 padUp,
                 memoryReusePolicy,
                 numCudaStreams)

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
   assert(not self.outputBuffer)
   assert(not self.outputTransposeBuffer)
   assert(not self.weightBuffer)
   assert(not self.weightTransposeBuffer)
end

function SpatialConvolutionFBFFTGemm:prepareSizeAndBuffers(i, w, o, metaData)
   return self:prepareFBFFTGemmSizeAndBuffers(i, w, o, metaData, metaData.pass)
end

--[[
   Update output
--]]
function SpatialConvolutionFBFFTGemm:updateOutputFFTImpl(input, reuseList)
   assert(torch.type(input) == 'torch.CudaTensor', "CUDA support only!")

   local metaData = {}
   metaData.pass = nn.SpatialConvolutionFFT.ForwardFFTPass

   local commonSize =
      self:prepareSizeAndBuffers(input, self.weight, self.output, metaData)

   local cublasBatchDims = 2
   -- 2D convolutions on 4D tensors atm
   assert(#input:size() == cublasBatchDims + 2)

   local inputFFTStream = 1
   local weightFFTStream = 2
   local gemmStream = 3
   assert(cutorch.getNumStreams() >= 3)

   -- Synchronize all streams on SESE, change when we have a proper DAG impl
   -- ##############################################
   cutorch.streamBarrier(self.allStreams)

   local fftWrapper = nn.FFTWrapper(self.fftImplementation)
   -- 1. FFT + transpose input and weights
   if not reuseList or
      not reuseList:contains(
         nn.SpatialConvolutionFFT.CuFFTInputTransposeBufferType)
   then
      cutorch.setStream(inputFFTStream)
      local fftWrapperPadded = nn.FFTWrapper(
         self.fftImplementation, self.padLeft, self.padUp)
      fftWrapperPadded:fftTranspose(input,
                                    self.inputBuffer,
                                    self.inputTransposeBuffer,
                                    cublasBatchDims,
                                    1, -- handle
                                    inputFFTStream -- stream
                                    )
   end

   if not reuseList or
      not reuseList:contains(
         nn.SpatialConvolutionFFT.CuFFTWeightTransposeBufferType)
   then
      cutorch.setStream(weightFFTStream)
      fftWrapper:fftTranspose(self.weight,
                              self.weightBuffer,
                              self.weightTransposeBuffer,
                              cublasBatchDims,
                              2, -- handle
                              weightFFTStream -- stream
                              )
   end

   -- 2. CGEMM on transposed tensors
   -- This call uses all the handles and streams available
   -- CuBLAS is column major and computes C' = B' * A'
   local useBatchedMM = (commonSize[3] * commonSize[4] >= 128)
   local cublasWrapper = nn.CuBLASWrapper()
   local norm = self:getNormalizationFactor(commonSize, input)

   if not useBatchedMM then
      -- Synchronize all streams: iterated GEMMS use all available streams
      -- ##############################################
      cutorch.streamBarrier(self.allStreams)

      -- a. multiple GEMMs on multiple streams
      cublasWrapper:matmultComplex(self.inputTransposeBuffer,
                                   self.weightTransposeBuffer,
                                   self.outputTransposeBuffer,
                                   {0, 1}, -- iterDims == 2
                                   { },    -- cublasBatchDims
                                   'n',
                                   'c',
                                   1.0 / norm)

      -- Synchronize all streams: iterated GEMMS use all available streams
      -- ##############################################
      cutorch.streamBarrier(self.allStreams)
   else
      -- stream must match the IFFT stream for sync without waiting
      -- explicitly
      cutorch.setStream(gemmStream)
      cutorch.streamWaitFor(gemmStream, {inputFFTStream, weightFFTStream})
      cublasWrapper:matmultComplex(self.inputTransposeBuffer,
                                   self.weightTransposeBuffer,
                                   self.outputTransposeBuffer,
                                   {},     -- iterDims
                                   {0, 1}, -- cublasBatchDims == 2
                                   'n',
                                   'c',
                                   1.0 / norm)
   end

   -- 3. transpose + IFFT output
   cutorch.setStream(gemmStream)
   fftWrapper:transposeIFFT(self.output,
                            self.outputBuffer,
                            self.outputTransposeBuffer,
                            cublasBatchDims,
                            1, -- handle
                            gemmStream -- stream
                            )

   -- ##############################################
   cutorch.streamBarrier(self.allStreams)

   -- 4. Finally, bias update
   cutorch.setStream(gemmStream)
   ConvolutionBiasFFI.updateOutputBiasFFI(
      cutorch._state, self.output:cdata(), self.bias:cdata())

   -- Synchronize all streams on SESE, change when we have a proper DAG impl
   -- ##############################################
   cutorch.streamBarrier(self.allStreams)

   return self.output
end

--[[
   Update input gradients
--]]


function SpatialConvolutionFBFFTGemm:updateGradInputFFTImpl(
      input, gradOutput, reuseList)
   assert(torch.type(input) == 'torch.CudaTensor', "CUDA support only!")

   local metaData = {}
   metaData.pass = nn.SpatialConvolutionFFT.BackwardFFTPass

   local commonSize =
      self:prepareSizeAndBuffers(input, self.weight, gradOutput, metaData)

   local cublasBatchDims = 2
   -- 2D convolutions on 4D tensors atm
   assert(#input:size() == cublasBatchDims + 2)

   local weightFFTStream = 1
   local gradOutputFFTStream = 2
   local gemmStream = 3
   assert(cutorch.getNumStreams() >= 3)

   -- Synchronize all streams on SESE, change when we have a proper DAG impl
   -- ##############################################
   cutorch.streamBarrier(self.allStreams)

   local fftWrapper = nn.FFTWrapper(self.fftImplementation)

   -- 1. FFT + transpose gradOutput and weights
   if not reuseList or
      not reuseList:contains(
         nn.SpatialConvolutionFFT.CuFFTOutputTransposeBufferType)
   then
      cutorch.setStream(gradOutputFFTStream)
      fftWrapper:fftTranspose(gradOutput,
                              self.outputBuffer,
                              self.outputTransposeBuffer,
                              cublasBatchDims,
                              1, -- handle
                              gradOutputFFTStream -- stream
                              )
   end

   if (not reuseList or
      not reuseList:contains(
         nn.SpatialConvolutionFFT.CuFFTWeightTransposeBufferType)) and
      not self.memoryReusePolicy:contains(
         nn.SpatialConvolutionFFT.memoryReuseWeight)
   then
      -- TODO: fix this: transpose changes the TH metadata post buffer
      -- get/put which screws up the tensor
      cutorch.setStream(weightFFTStream)
      fftWrapper:fftTranspose(self.weight,
                              self.weightBuffer,
                              self.weightTransposeBuffer,
                              cublasBatchDims,
                              2, -- handle
                              weightFFTStream -- stream
                              )
   end

   -- 2. CGEMM on transposed tensors
   -- This call uses all the handles and streams available
   -- CuBLAS is column major and computes C' = B' * A'
   local useBatchedMM = (commonSize[3] * commonSize[4] >= 128)
   local cublasWrapper = nn.CuBLASWrapper()
   local norm = self:getNormalizationFactor(commonSize, gradOutput)
   if not useBatchedMM then
      -- Synchronize all streams: iterated GEMMS use all available streams
      -- ##############################################
      cutorch.streamBarrier(self.allStreams)

      cublasWrapper:matmultComplex(self.outputTransposeBuffer,
                                   self.weightTransposeBuffer,
                                   self.inputTransposeBuffer,
                                   {0, 1}, -- iterDims == 2
                                   { },    -- cublasBatchDims
                                   'n',
                                   'n',
                                   1.0 / norm)

      -- Synchronize all streams: iterated GEMMS use all available streams
      -- ##############################################
      cutorch.streamBarrier(self.allStreams)
   else
      -- stream must match the IFFT stream for sync without waiting
      -- explicitly
      cutorch.setStream(gemmStream)
      cutorch.streamWaitFor(gemmStream, {weightFFTStream, gradOutputFFTStream})

      cublasWrapper:matmultComplex(self.outputTransposeBuffer,
                                   self.weightTransposeBuffer,
                                   self.inputTransposeBuffer,
                                   { },    -- iterDims
                                   {0, 1}, -- cublasBatchDims == 2
                                   'n',
                                   'n',
                                   1.0 / norm)
   end

   -- 3. transpose + IFFT gradInput
   cutorch.setStream(gemmStream)

   local fftWrapperPadded = nn.FFTWrapper(
      self.fftImplementation, self.padLeft, self.padUp)
   fftWrapperPadded:transposeIFFT(self.gradInput,
                                  self.inputBuffer,
                                  self.inputTransposeBuffer,
                                  cublasBatchDims,
                                  1, -- handle
                                  gemmStream -- stream
                                  )

   -- ##############################################
   cutorch.streamBarrier(self.allStreams)

   return self.gradInput
end


--[[
   Accumulate weight gradients
--]]
function SpatialConvolutionFBFFTGemm:accGradParametersFFTImpl(
      input, gradOutput, scale, reuseList)
   assert(torch.type(input) == 'torch.CudaTensor', "CUDA support only!")
   scale = scale or 1

   local metaData = {}
   metaData.pass = nn.SpatialConvolutionFFT.AccGradientFFTPass

   local commonSize =
      self:prepareSizeAndBuffers(input, self.gradWeight, gradOutput, metaData)

   local cublasBatchDims = 2
   -- 2D convolutions on 4D tensors atm
   assert(#input:size() == cublasBatchDims + 2)

   local inputFFTStream = 1
   local gradOutputFFTStream = 2
   local gradBiasFFTStream = 3
   local gemmStream = 4
   assert(cutorch.getNumStreams() >= gemmStream)

   -- Synchronize all streams on SESE, change when we have a proper DAG impl
   -- ##############################################
   cutorch.streamBarrier(self.allStreams)

   local fftWrapper = nn.FFTWrapper(self.fftImplementation)

   -- 0. gradBIas update is independent
   cutorch.setStream(gradBiasFFTStream)
   ConvolutionBiasFFI.accGradParametersBiasFFI(
      cutorch._state, gradOutput:cdata(), self.gradBias:cdata(), scale)

   -- 1. FFT + transpose gradOutput and weights
   if (not reuseList or
          not reuseList:contains(
             nn.SpatialConvolutionFFT.CuFFTOutputTransposeBufferType)) and
      not self.memoryReusePolicy:contains(
         nn.SpatialConvolutionFFT.memoryReuseOutput)
   then
      -- TODO: fix this: transpose changes the TH metadata post buffer
      -- get/put which screws up the tensor
      cutorch.setStream(gradOutputFFTStream)
      fftWrapper:fftTranspose(gradOutput,
                              self.outputBuffer,
                              self.outputTransposeBuffer,
                              cublasBatchDims,
                              1,
                              gradOutputFFTStream)
   end

   if (not reuseList or
          not reuseList:contains(
             nn.SpatialConvolutionFFT.CuFFTInputTransposeBufferType)) and
      not self.memoryReusePolicy:contains(
         nn.SpatialConvolutionFFT.memoryReuseInput)
   then
      cutorch.setStream(inputFFTStream)
      local fftWrapperPadded = nn.FFTWrapper(
         self.fftImplementation, self.padLeft, self.padUp)
      fftWrapperPadded:fftTranspose(input,
                                    self.inputBuffer,
                                    self.inputTransposeBuffer,
                                    cublasBatchDims,
                                    2,
                                    inputFFTStream)
   end

   -- 2. CGEMM on transposed tensors
   -- This call uses all the handles and streams available
   -- CuBLAS is column major and computes C' = B' * A'
   local useBatchedMM = (commonSize[3] * commonSize[4] >= 128)
   local cublasWrapper = nn.CuBLASWrapper()
   local norm = self:getNormalizationFactor(commonSize, gradOutput)
   if not useBatchedMM then
      -- Synchronize all streams: iterated GEMMS use all available streams
      -- ##############################################
      cutorch.streamBarrier(self.allStreams)

      cublasWrapper:matmultComplex(self.outputTransposeBuffer,
                                   self.inputTransposeBuffer,
                                   self.weightTransposeBuffer,
                                   {0, 1}, -- iterDims == 2
                                   { },    -- cublasBatchDims
                                   'c',
                                   'n',
                                   (1.0 * scale) / norm)

      -- Synchronize all streams: iterated GEMMS use all available streams
      -- ##############################################
      cutorch.streamBarrier(self.allStreams)
   else
      -- stream must match the IFFT stream for sync without waiting
      -- explicitly
      cutorch.setStream(gemmStream)
      cutorch.streamWaitFor(gemmStream, {inputFFTStream, gradOutputFFTStream})

      cublasWrapper:matmultComplex(self.outputTransposeBuffer,
                                   self.inputTransposeBuffer,
                                   self.weightTransposeBuffer,
                                   { },    -- iterDims
                                   {0, 1}, -- cublasBatchDims == 2
                                   'c',
                                   'n',
                                   (1.0 * scale) / norm)
   end

   -- 3. transpose + IFFT gradInput
   cutorch.setStream(gemmStream)
   fftWrapper:transposeIFFT(self.gradWeight,
                            self.weightBuffer,
                            self.weightTransposeBuffer,
                            cublasBatchDims,
                            1,          -- handle
                            gemmStream -- stream
                            )

   -- ##############################################
   cutorch.streamBarrier(self.allStreams)
end


--[[
   -- Buffer creation and reuse given a size and a pass.
   -- Different passes use different tensors as the 'output of the pass'.
   --   nn.SpatialConvolutionFFT.ForwardFFTPass -> output
   --   nn.SpatialConvolutionFFT.BackwardFFTPass -> input
   --   nn.SpatialConvolutionFFT.AccGradientFFTPass -> weight
   -- The buffers corresponding to the tensors that is the 'output of the pass'
   -- must be properly transposed in order for the CGemm call to be consistent.
   -- This is a simple metadata transposition, might as well construct properly.
--]]
function SpatialConvolutionFBFFTGemm:prepareBuffers(commonSize, pass, metaData)
   assert(commonSize and pass and self.fftImplementation)
   assert(torch.type(metaData) == 'table', torch.type(metaData))

   if not parent.prepareBuffers(self, commonSize, pass, metaData)
   then
      return false
   end

   local bufferSizesO = torch.LongStorage({
         commonSize[1], self.nOutputPlane, commonSize[3], commonSize[4]})
   local bufferSizesW = torch.LongStorage({
         self.nOutputPlane, self.nInputPlane, commonSize[3], commonSize[4]})

   self.inputTransposeBuffer = self:getBuffer(
      nn.SpatialConvolutionFFT.CuFFTInputTransposeBufferType,
      commonSize,
      true,
      metaData)
   self.outputTransposeBuffer = self:getBuffer(
      nn.SpatialConvolutionFFT.CuFFTOutputTransposeBufferType,
      bufferSizesO,
      true,
      metaData)
   self.weightTransposeBuffer = self:getBuffer(
      nn.SpatialConvolutionFFT.CuFFTWeightTransposeBufferType,
      bufferSizesW,
      true,
      metaData)

   if self.inputTransposeBuffer and
      self.outputTransposeBuffer and
      self.weightTransposeBuffer then
         return true
   end

   print('Not enough memory for FBFFTGemm buffers, need to fall back')

   -- TODO: From here on, we should failsafe to another SpatialConvolution
   self:cleanupBuffers()

   assert(false, 'Out of memory!')
end

function SpatialConvolutionFBFFTGemm:cleanupBuffers()
   parent.cleanupBuffers(self)

   -- Kill local references to global buffers
   self.inputTransposeBuffer = nil
   self.outputTransposeBuffer = nil
   self.weightTransposeBuffer = nil
end


function SpatialConvolutionFBFFTGemm:getBufferKey(
      BufferType, bufferSizes, metaData)
   assert(torch.type(bufferSizes) == 'torch.LongStorage',
          torch.type(bufferSizes))
   assert(torch.type(metaData) == 'table', torch.type(metaData))

   -- If no reuse, we hit into the buffers discrimianted by device and
   -- BufferType. These buffers are shared with all FFT convolution modules
   -- and do not allow reuse for long dependences (i.e. only gradOutput can
   -- only be reused from a supporting backward implementation)
   if self.memoryReusePolicy:contains(nn.SpatialConvolutionFFT.memoryReuseNone)
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
      -- In cufft mode, the transposed complex buffers are reused
      if (metaData.pass == nn.SpatialConvolutionFFT.ForwardFFTPass and
             BufferType ==
             nn.SpatialConvolutionFFT.CuFFTOutputTransposeBufferType) or
         (metaData.pass == nn.SpatialConvolutionFFT.BackwardFFTPass and
             BufferType ==
             nn.SpatialConvolutionFFT.CuFFTInputTransposeBufferType) or
         (metaData.pass == nn.SpatialConvolutionFFT.AccGradientFFTPass and
             BufferType ==
             nn.SpatialConvolutionFFT.CuFFTWeightTransposeBufferType)
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

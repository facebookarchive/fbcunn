-- Copyright 2004-present Facebook. All Rights Reserved.

require 'cudnn'
local List = require 'pl.List'
local thrift = require('fb.thrift')
local ffi = require 'ffi'
local lib_name = 'torch_fb_fbcunn_convolution_bias'
local lib_path = package.searchpath(lib_name, package.cpath)
local ConvolutionBiasFFI = ffi.load(lib_path and lib_path or lib_name)

--[[
   Actual module
--]]
local SpatialConvolutionCuFFT, parent =
   torch.class('nn.SpatialConvolutionCuFFT', 'nn.SpatialConvolutionFFT')

function SpatialConvolutionCuFFT:__init(nInputPlane,
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

   parent.fftImplementation = 'cufft'

   assert(self.padUp == 0 and
             self.padDown == 0 and
             self.padLeft == 0 and
             self.padRight == 0, "cufft does not support implicit padding!")

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

   -- CuFFT plans
   assert(not self.cufftPlanInputFFT)
   assert(not self.cufftPlanWeightFFT)
   assert(not self.cufftPlanOutputFFT)
   assert(not self.cufftPlanInputIFFT)
   assert(not self.cufftPlanWeightIFFT)
   assert(not self.cufftPlanOutputIFFT)
end

--[[
   Helper function to perform explicit padding
   In the case of cufft, padding must be explicit with zeros on the
   inputs of the algorithm. fbfft does not need this.
--]]
function SpatialConvolutionCuFFT:isOutputOfPass(pass, tensor)
   assert(pass == nn.SpatialConvolutionFFT.ForwardFFTPass or
             pass == nn.SpatialConvolutionFFT.BackwardFFTPass or
             pass == nn.SpatialConvolutionFFT.AccGradientFFTPass)
   if pass == nn.SpatialConvolutionFFT.ForwardFFTPass and
         tensor == self.output
   then
      return true
   end
   if pass == nn.SpatialConvolutionFFT.BackwardFFTPass and
         tensor == self.gradInput
   then
      return true
   end
   if pass == nn.SpatialConvolutionFFT.AccGradientFFTPass and
         tensor == self.gradWeight
   then
      return true
   end
   return false
end

function SpatialConvolutionCuFFT:fftPadding(tensor, pass, inputTensor)
   -- Always input, weight, output
   local tensorList = {}
   local paddedList = {}
   if pass == nn.SpatialConvolutionFFT.ForwardFFTPass then
      tensorList = {tensor, self.weight, self.output}
      paddedList = {self.inputPadded, self.weightPadded, self.outputPadded}
   elseif pass == nn.SpatialConvolutionFFT.BackwardFFTPass then
      tensorList = {self.gradInput, self.weight, tensor}
      paddedList = {self.inputPadded, self.weightPadded, self.outputPadded}
   elseif pass == nn.SpatialConvolutionFFT.AccGradientFFTPass then
      tensorList = {inputTensor, self.gradWeight, tensor}
      paddedList = {self.inputPadded, self.weightPadded, self.outputPadded}
   end

   for ind = 1, #tensorList do
      -- If we have a non empty padded tensor
      if paddedList[ind] and paddedList[ind]:nElement() > 0 then
         local _orig   = tensorList[ind]
         local padded = paddedList[ind]
         if not self:isOutputOfPass(pass, tensorList[ind]) then
            local sizes = tensorList[ind]:size()
            local paddedSizes = paddedList[ind]:size()
            -- resize messes up strides, I want a fortran subarray here,
            -- do it manually
            padded:set(padded:storage(),
                       padded:storageOffset(),
                       sizes,
                       padded:stride())
            padded:copy(tensorList[ind])
            -- make tensor full again, it is now contiguous and zero padded
            padded:set(padded:storage(),
                       padded:storageOffset(),
                       paddedSizes, padded:stride())
         end
      end
   end

   -- swap original and padded tensors to be transparent for the
   -- convolution pass
   if pass == nn.SpatialConvolutionFFT.ForwardFFTPass then
      if self.inputPadded and self.inputPadded:nElement() > 0 then
         tensor, self.inputPadded = self.inputPadded, tensor
      end
      if self.weightPadded and self.weightPadded:nElement() > 0 then
         self.weight, self.weightPadded = self.weightPadded, self.weight
      end
      if self.outputPadded and self.outputPadded:nElement() > 0 then
         self.output, self.outputPadded = self.outputPadded, self.output
      end
   elseif pass == nn.SpatialConvolutionFFT.BackwardFFTPass then
      if self.inputPadded and self.inputPadded:nElement() > 0 then
         self.gradInput, self.inputPadded = self.inputPadded, self.gradInput
      end
      if self.weightPadded and self.weightPadded:nElement() > 0 then
         self.weight, self.weightPadded = self.weightPadded, self.weight
      end
      if self.outputPadded and self.outputPadded:nElement() > 0 then
         tensor, self.outputPadded = self.outputPadded, tensor
      end
   elseif pass == nn.SpatialConvolutionFFT.AccGradientFFTPass then
      if self.inputPadded and self.inputPadded:nElement() > 0 then
         inputTensor, self.inputPadded = self.inputPadded, inputTensor
      end
      if self.weightPadded and self.weightPadded:nElement() > 0 then
         self.gradWeight, self.weightPadded = self.weightPadded, self.gradWeight
      end
      if self.outputPadded and self.outputPadded:nElement() > 0 then
         tensor, self.outputPadded = self.outputPadded, tensor
      end
   end

   return tensor, inputTensor
end


--[[
   Helper function to undo padding
   In the case of cufft, padding must be explicit with zeros on the
   inputs of the algorithm. fbfft does not need this.
--]]
function SpatialConvolutionCuFFT:fftUnpadding(tensor, pass, inputTensor)
   -- Always input, weight, output
   local tensorList = {}
   local paddedList = {}
   -- Here the paddedList and tensorList are reversed compared to fftPadding
   -- Only true for those tensors that are actually padded (i.e. self.
   -- inputPadded both non nil and not empty)
   if pass == nn.SpatialConvolutionFFT.ForwardFFTPass then
      paddedList = {tensor, self.weight, self.output}
      tensorList = {self.inputPadded, self.weightPadded, self.outputPadded}
   elseif pass == nn.SpatialConvolutionFFT.BackwardFFTPass then
      paddedList = {self.gradInput, self.weight, tensor}
      tensorList = {self.inputPadded, self.weightPadded, self.outputPadded}
   elseif pass == nn.SpatialConvolutionFFT.AccGradientFFTPass then
      paddedList = {inputTensor, self.gradWeight, tensor}
      tensorList = {self.inputPadded, self.weightPadded, self.outputPadded}
   end

   for ind = 1, #tensorList do
      -- If we have a non-empty padded tensor
      if tensorList[ind] and tensorList[ind]:nElement() > 0 then
         local orig   = tensorList[ind]
         local padded = paddedList[ind]
         if self:isOutputOfPass(pass, paddedList[ind]) then
            local sizes = tensorList[ind]:size()
            local paddedSizes = paddedList[ind]:size()
            -- resize messes up strides, I want a fortran subarray here,
            -- do it manually
            padded:set(padded:storage(),
                       padded:storageOffset(),
                       sizes,
                       padded:stride())
            orig:copy(padded)
            -- make tensor full again, it is now contiguous and zero padded
            padded:set(padded:storage(),
                       padded:storageOffset(),
                       paddedSizes,
                       padded:stride())
         end
      end
   end

   -- swap original and padded tensors to be transparent for the
   -- convolution pass
   if pass == nn.SpatialConvolutionFFT.ForwardFFTPass then
      if self.inputPadded and self.inputPadded:nElement() > 0 then
         tensor, self.inputPadded = self.inputPadded, tensor
      end
      if self.weightPadded and self.weightPadded:nElement() > 0 then
         self.weight, self.weightPadded = self.weightPadded, self.weight
      end
      if self.outputPadded and self.outputPadded:nElement() > 0 then
         self.output, self.outputPadded = self.outputPadded, self.output
      end
   elseif pass == nn.SpatialConvolutionFFT.BackwardFFTPass then
      if self.inputPadded and self.inputPadded:nElement() > 0 then
         self.gradInput, self.inputPadded = self.inputPadded, self.gradInput
      end
      if self.weightPadded and self.weightPadded:nElement() > 0 then
         self.weight, self.weightPadded = self.weightPadded, self.weight
      end
      if self.outputPadded and self.outputPadded:nElement() > 0 then
         tensor, self.outputPadded = self.outputPadded, tensor
      end
   elseif pass == nn.SpatialConvolutionFFT.AccGradientFFTPass then
      if self.inputPadded and self.inputPadded:nElement() > 0 then
         inputTensor, self.inputPadded = self.inputPadded, inputTensor
      end
      if self.weightPadded and self.weightPadded:nElement() > 0 then
         self.gradWeight, self.weightPadded = self.weightPadded, self.gradWeight
      end
      if self.outputPadded and self.outputPadded:nElement() > 0 then
         tensor, self.outputPadded = self.outputPadded, tensor
      end
   end

   return tensor, inputTensor
end

function SpatialConvolutionCuFFT:prepareSizeAndBuffers(i, w, o, metaData)
   return self:prepareCuFFTSizeAndBuffers(i, w, o, metaData, metaData.pass)
end

--[[
   Update output
--]]
function SpatialConvolutionCuFFT:updateOutputFFTImpl(input, reuseList)
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
   -- In cufft mode, we have explicit padding tensors
   input = self:fftPadding(input, nn.SpatialConvolutionFFT.ForwardFFTPass)
   -- Padding / unpadding perform copies on default stream, synchronize all
   -- ##############################################
   cutorch.streamBarrier(self.allStreams)

   -- 1. FFT + transpose input and weights
   if not reuseList or
      not reuseList:contains(
         nn.SpatialConvolutionFFT.CuFFTInputTransposeBufferType)
   then
      cutorch.setStream(inputFFTStream)
      self.cufftPlanInputFFT =
         fftWrapper:fftTranspose(input,
                                 self.inputBuffer,
                                 self.inputTransposeBuffer,
                                 cublasBatchDims,
                                 1, -- handle
                                 inputFFTStream, -- stream
                                 self.cufftPlanInputFFT)
   end

   if not reuseList or
      not reuseList:contains(
         nn.SpatialConvolutionFFT.CuFFTWeightTransposeBufferType)
   then
      cutorch.setStream(weightFFTStream)
      self.cufftPlanWeightFFT =
         fftWrapper:fftTranspose(self.weight,
                                 self.weightBuffer,
                                 self.weightTransposeBuffer,
                                 cublasBatchDims,
                                 2, -- handle
                                 weightFFTStream, -- stream
                                 self.cufftPlanWeightFFT)
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
   self.cufftPlanOutputIFFT =
      fftWrapper:transposeIFFT(self.output,
                               self.outputBuffer,
                               self.outputTransposeBuffer,
                               cublasBatchDims,
                               1, -- handle
                               gemmStream, -- stream
                               self.cufftPlanOutputIFFT)

   -- ##############################################
   -- Padding / unpadding perform copies on default stream, synchronize all
   cutorch.streamBarrier(self.allStreams)

   -- 4. If cufft, needs resize
   self:fftUnpadding(input, nn.SpatialConvolutionFFT.ForwardFFTPass)

   -- Synchronize all: Padding / unpadding perform copies on default stream
   -- ##############################################
   cutorch.streamBarrier(self.allStreams)

   -- 5. Finally, bias update
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


function SpatialConvolutionCuFFT:updateGradInputFFTImpl(
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
   -- If cufft, we may have padding tensors into which to copy the data
   gradOutput = self:fftPadding(gradOutput,
                                nn.SpatialConvolutionFFT.BackwardFFTPass)
   -- Padding / unpadding perform copies on default stream, synchronize all
   -- ##############################################
   cutorch.streamBarrier(self.allStreams)

   -- 1. FFT + transpose gradOutput and weights
   if not reuseList or
      not reuseList:contains(
         nn.SpatialConvolutionFFT.CuFFTOutputTransposeBufferType)
   then
      cutorch.setStream(gradOutputFFTStream)
      self.cufftPlanOutputFFT =
         fftWrapper:fftTranspose(gradOutput,
                                 self.outputBuffer,
                                 self.outputTransposeBuffer,
                                 cublasBatchDims,
                                 1, -- handle
                                 gradOutputFFTStream, -- stream
                                 self.cufftPlanOutputFFT)
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
      self.cufftPlanWeightFFT =
         fftWrapper:fftTranspose(self.weight,
                                 self.weightBuffer,
                                 self.weightTransposeBuffer,
                                 cublasBatchDims,
                                 2, -- handle
                                 weightFFTStream, -- stream
                                 self.cufftPlanWeightFFT)
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
   self.cufftPlanInputIFFT =
      fftWrapper:transposeIFFT(self.gradInput,
                               self.inputBuffer,
                               self.inputTransposeBuffer,
                               cublasBatchDims,
                               1, -- handle
                               gemmStream, -- stream
                               self.cufftPlanInputIFFT)

   -- ##############################################
   -- Padding / unpadding perform copies on default stream, synchronize all
   cutorch.streamBarrier(self.allStreams)

   -- 4. If cufft, needs resize
   self:fftUnpadding(gradOutput, nn.SpatialConvolutionFFT.BackwardFFTPass)

   -- Padding / unpadding perform copies on default stream, synchronize all
   -- ##############################################
   cutorch.streamBarrier(self.allStreams)

   -- 5. No bias operation

   return self.gradInput
end


--[[
   Accumulate weight gradients
--]]
function SpatialConvolutionCuFFT:accGradParametersFFTImpl(
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
   -- If cufft, we may have padding tensors into which to copy the data
   gradOutput, input = self:fftPadding(
      gradOutput,  nn.SpatialConvolutionFFT.AccGradientFFTPass, input)
   assert(self.gradWeight:size(3) == commonSize[3])
   assert(self.gradWeight:size(4) == commonSize[4])

   -- Padding / unpadding perform copies on default stream, synchronize all
   -- ##############################################
   cutorch.streamBarrier(self.allStreams)

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
      self.cufftPlanOutputFFT =
         fftWrapper:fftTranspose(gradOutput,
                                 self.outputBuffer,
                                 self.outputTransposeBuffer,
                                 cublasBatchDims,
                                 1,
                                 gradOutputFFTStream,
                                 self.cufftPlanOutputFFT)
   end

   if (not reuseList or
          not reuseList:contains(
             nn.SpatialConvolutionFFT.CuFFTInputTransposeBufferType)) and
      not self.memoryReusePolicy:contains(
         nn.SpatialConvolutionFFT.memoryReuseInput)
   then
      cutorch.setStream(inputFFTStream)
      self.cufftPlanInputFFT =
         fftWrapper:fftTranspose(input,
                                 self.inputBuffer,
                                 self.inputTransposeBuffer,
                                 cublasBatchDims,
                                 2,
                                 inputFFTStream,
                                 self.cufftPlanInputFFT)
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
   self.cufftPlanWeightIFFT =
      fftWrapper:transposeIFFT(self.gradWeight,
                               self.weightBuffer,
                               self.weightTransposeBuffer,
                               cublasBatchDims,
                               1,          -- handle
                               gemmStream, -- stream
                               self.cufftPlanWeightIFFT)

   -- ##############################################
   -- Padding / unpadding perform copies on default stream, synchronize all
   cutorch.streamBarrier(self.allStreams)

   -- 4. If cufft, needs resize
   self:fftUnpadding(
      gradOutput, nn.SpatialConvolutionFFT.AccGradientFFTPass, input)
   assert(self.gradWeight:size(3) == self.kH)
   assert(self.gradWeight:size(4) == self.kW)

   -- Padding / unpadding perform copies on default stream, synchronize all
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
function SpatialConvolutionCuFFT:prepareBuffers(commonSize, pass, metaData)
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

   self.inputPadded = self:getBuffer(
      nn.SpatialConvolutionFFT.CuFFTPaddedInputBuffer,
      commonSize,
      false,
      metaData)
   self.outputPadded = self:getBuffer(
      nn.SpatialConvolutionFFT.CuFFTPaddedOutputBuffer,
      bufferSizesO,
      false,
      metaData)
   self.weightPadded = self:getBuffer(
      nn.SpatialConvolutionFFT.CuFFTPaddedWeightBuffer,
      bufferSizesW,
      false,
      metaData)

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

   if self.inputTransposeBuffer and self.inputPadded and
      self.outputTransposeBuffer and self.outputPadded and
      self.weightTransposeBuffer and self.weightPadded then
         return true
   end

   print('Not enough memory for CuFFT buffers, need to fall back')

   -- TODO: From here on, we should failsafe to another SpatialConvolution
   self:cleanupBuffers()

   return false
end

function SpatialConvolutionCuFFT:cleanupBuffers()
   parent.cleanupBuffers(self)

   -- Kill cufft plans references to trigger GC
   self.cufftPlanInputFFT = nil
   self.cufftPlanWeightFFT = nil
   self.cufftPlanOutputFFT = nil
   self.cufftPlanInputIFFT = nil
   self.cufftPlanWeightIFFT = nil
   self.cufftPlanOutputIFFT = nil

   -- Kill local references to global buffers
   self.inputTransposeBuffer = nil
   self.inputPadded = nil
   self.outputTransposeBuffer = nil
   self.outputPadded = nil
   self.weightTransposeBuffer = nil
   self.weightPadded = nil
end


 -- TODO: CuFFT is more flexible to allow for arbitrary FFT interpolation sizes.
 -- When writing the autotuner, it is easy to get different interpolation sizes
 -- for the FFTs in the 3 passes, perform best.
 -- For correction of reuse, reuse should only work if interpolation sizes are
 -- the same between 2 passes.
 -- In practice this means supporting real producer / consumer semantics in the
 -- tag space. In particular we need to match any read to a unique write and
 -- ensure they occur in the proper order.
 -- For instance, there is no reason that updateGradInput occurs before
 -- accGradParameters so we need to ensure the first one writes gradOutput and
 -- the second one reads it
function SpatialConvolutionCuFFT:getBufferKey(BufferType, bufferSizes, metaData)
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

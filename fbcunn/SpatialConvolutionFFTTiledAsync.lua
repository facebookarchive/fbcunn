-- Copyright 2004-present Facebook. All Rights Reserved.

require 'cudnn'
local List = require 'pl.List'
local ffi = require 'ffi'

local lib_name = 'torch_fb_fbcunn_mm'
local lib_path = package.searchpath(lib_name, package.cpath)
local FBMMFFI = ffi.load(lib_path and lib_path or lib_name)

local lib_name = 'torch_fb_fbcunn_convolution_bias'
local lib_path = package.searchpath(lib_name, package.cpath)
local ConvolutionBiasFFI = ffi.load(lib_path and lib_path or lib_name)

local function errorIf(cond, msg)
   if cond then
      error(msg)
   end
end

local function errorIfNot(cond, msg)
   errorIf(not cond, msg)
end

local function equalsTiledTensorDescriptor(td1, td2)
   local res = true
   if td1.tileSizeH ~= td2.tileSizeH then
      res = res and false
   end
   if td1.tileSizeW ~= td2.tileSizeW then
      res = res and false
   end
   if td1.tileIndexH ~= td2.tileIndexH then
      res = res and false
   end
   if td1.tileIndexW ~= td2.tileIndexW then
      res = res and false
   end
   if td1.tensorSizeH ~= td2.tensorSizeH then
      res = res and false
   end
   if td1.tensorSizeW ~= td2.tensorSizeW then
      res = res and false
   end
   if td1.padUp ~= td2.padUp then
      res = res and false
   end
   if td1.padLeft ~= td2.padLeft then
      res = res and false
   end
   if td1.tensor:storage() ~= td2.tensor:storage() then
      res = res and false
   end
   if td1.tensor:storageOffset() ~= td2.tensor:storageOffset() then
      res = res and false
   end
   return res
end


------------------------------------------------------------------------------
--   Actual Module
------------------------------------------------------------------------------
local SpatialConvolutionFFTTiledAsync, parent =
   torch.class('nn.SpatialConvolutionFFTTiledAsync',
               'nn.SpatialConvolutionFFTTiled')

function SpatialConvolutionFFTTiledAsync:__init(nInputPlane,
                                                nOutputPlane,
                                                kW,
                                                kH,
                                                dW,
                                                dH,
                                                padLeft,
                                                padUp,
                                                tileSizeW,
                                                tileSizeH,
                                                memoryReusePolicy,
                                                numCudaStreams)
   parent.__init(self,
                 nInputPlane,
                 nOutputPlane,
                 kW,
                 kH,
                 dW,
                 dH,
                 padLeft,
                 padUp,
                 tileSizeW,
                 tileSizeH,
                 memoryReusePolicy,
                 numCudaStreams)
end


function SpatialConvolutionFFTTiledAsync:instUpdateOutputFFTImpl(input)
   -- Make sure tiling information has been precomputed
   assert(self.inputTensorList)
   assert(self.outputTensorList)

   local currentStream = 1
   -- ##############################################
   cutorch.streamBarrier(self.allStreams)

   for i = 1, self.outputTensorList:len() do
      -- Assert consistency of tensor dimensions
      errorIfNot(#self.inputTensorList[i].tensor:size() == #input:size(),
                 "Tensor size mismatch: " ..
                    #self.inputTensorList[i].tensor:size() .. " vs " ..
                    #input:size())
      errorIfNot(#self.outputTensorList[i].tensor:size() == #self.output:size())

      -- Set padding for this tile which can be partial and on the boundary
      local savePadUp, savePadLeft, savePadDown, savePadRight =
         self:pushPadding(i, self.inputTensorList)

      local firstIteration = (i == 1)
      local reuseList = List{}
      if not firstIteration then
         -- Whatever the memory reuse policy, when tiling, we can reuse
         -- the computed FFT(weight), this is one of the points of tiling
         reuseList:append(self.FFTWeightBufferType)
      end
      local inputLocal = self.inputTensorList[i].tensor
      local outputLocal = self.outputTensorList[i].tensor
      local metaData = self.metaDataListUpdateOutput[i]
      local cublasBatchDims = 2
      -- 2D convolutions on 4D tensors atm
      assert(#inputLocal:size() == cublasBatchDims + 2)

      local commonSize = self:prepareSizeAndBuffers(
         inputLocal, self.weight, outputLocal, metaData)

      -- Run all under this currentStream
      cutorch.setStream(currentStream)
      local fftWrapperPadded = nn.FFTWrapper(
         self.fftImplementation, self.padLeft, self.padUp)
      fftWrapperPadded:fft(inputLocal, self.inputBuffer, cublasBatchDims)
      if not reuseList or not reuseList:contains(self.FFTWeightBufferType) then
         local fftWrapper = nn.FFTWrapper(self.fftImplementation)
         fftWrapper:fft(self.weight, self.weightBuffer, cublasBatchDims)
         -- Since we're running async, everyone must wait on my mighty buffers
         -- ##############################################
         cutorch.streamBarrier(self.allStreams)
      end
      local norm = self:getNormalizationFactor(commonSize, inputLocal)
      FBMMFFI.transposeMMFFI(cutorch._state,
                             self.inputBuffer:cdata(),
                             self.weightBuffer:cdata(),
                             self.outputBuffer:cdata(),
                             1.0 / norm,
                             false,
                             true,
                             false)
      local fftWrapper = nn.FFTWrapper(self.fftImplementation)
      fftWrapper:ffti(outputLocal, self.outputBuffer, cublasBatchDims)
      currentStream = currentStream % cutorch.getNumStreams() + 1

      -- Pop back saved padding values
      self.padUp, self.padLeft, self.padDown, self.padRight =
         savePadUp, savePadLeft, savePadDown, savePadRight
   end

   -- ##############################################
   cutorch.streamBarrier(self.allStreams)
   cutorch.setStream(1)
   ConvolutionBiasFFI.updateOutputBiasFFI(
      cutorch._state, self.output:cdata(), self.bias:cdata())
   -- ##############################################
   cutorch.streamBarrier(self.allStreams)

   return self.output
end


function SpatialConvolutionFFTTiledAsync:instUpdateGradInputFFTImpl(
      input, gradOutput)
   -- Make sure tiling information has been precomputed
   assert(self.gradInputTensorList)
   assert(self.gradOutputTensorList)

   local currentStream = 1
   -- ##############################################
   cutorch.streamBarrier(self.allStreams)

   for i = 1, self.gradInputTensorList:len() do
      -- Assert consistency of tensor dimensions
      errorIfNot(#self.gradInputTensorList[i].tensor:size() == #input:size(),
             "Tensor size mismatch: " ..
                #self.gradInputTensorList[i].tensor:size() ..
                " vs " .. #self.gradInput:size())
      errorIfNot(
         #self.gradOutputTensorList[i].tensor:size() == #gradOutput:size())

      -- Set padding for this tile which can be partial and on the boundary
      -- Need additional padding for circular symmetry in Fourier domain
      local savePadUp, savePadLeft, savePadDown, savePadRight =
         self:pushPaddingWithCircularSymmetry(i, self.tileSizeH, self.tileSizeW)

      local firstIteration = (i == 1)
      local reuseList = List{}
      if not firstIteration then
         -- Whatever the memory reuse policy, when tiling, we can reuse
         -- the computed FFT(weight), this is one of the points of tiling
         reuseList:append(self.FFTWeightBufferType)
      end

      local inputLocal = self.gradInputTensorList[i].tensor
      local outputLocal = self.gradOutputTensorList[i].tensor
      local metaData = self.metaDataListUpdateGradInput[i]
      local cublasBatchDims = 2
      -- 2D convolutions on 4D tensors atm
      assert(#inputLocal:size() == cublasBatchDims + 2)

      local commonSize = self:prepareSizeAndBuffers(
         inputLocal, self.weight, outputLocal, metaData)

      -- Run all under this currentStream
      cutorch.setStream(currentStream)
      local fftWrapper = nn.FFTWrapper(self.fftImplementation)
      fftWrapper:fft(outputLocal, self.outputBuffer, cublasBatchDims)
      if not reuseList or not reuseList:contains(self.FFTWeightBufferType) then
         local fftWrapper = nn.FFTWrapper(self.fftImplementation)
         fftWrapper:fft(self.weight, self.weightBuffer, cublasBatchDims)
         -- Since we're running async, everyone must wait on my mighty buffers
         -- ##############################################
         cutorch.streamBarrier(self.allStreams)
      end
      local norm = self:getNormalizationFactor(commonSize, outputLocal)
      FBMMFFI.transposeMMFFI(cutorch._state,
                             self.outputBuffer:cdata(),
                             self.weightBuffer:cdata(),
                             self.inputBuffer:cdata(),
                             1.0 / norm,
                             false,
                             false,
                             false)
      local fftWrapperPadded = nn.FFTWrapper(
         self.fftImplementation, self.padLeft, self.padUp)
      fftWrapperPadded:ffti(inputLocal, self.inputBuffer, cublasBatchDims)
      currentStream = currentStream % cutorch.getNumStreams() + 1

      -- Pop back saved padding values
      self.padUp, self.padLeft, self.padDown, self.padRight =
         savePadUp, savePadLeft, savePadDown, savePadRight
   end

   -- ##############################################
   cutorch.streamBarrier(self.allStreams)

   return self.gradInput
end


function SpatialConvolutionFFTTiledAsync:instAccGradParametersFFTImpl(
      input, gradOutput, scale)
   -- Make sure tiling information has been precomputed
   assert(self.inputTensorList2)
   assert(self.gradOutputTensorList2)

   -- At this point tiles / metadata for buffer management / reuse are available
   local previousStream
   local currentStream = 1
   -- ##############################################
   cutorch.streamBarrier(self.allStreams)

   -- Run ahead
   cutorch.setStream(currentStream)
   ConvolutionBiasFFI.accGradParametersBiasFFI(
      cutorch._state, gradOutput:cdata(), self.gradBias:cdata(), scale)
   previousStream, currentStream =
      currentStream, currentStream % cutorch.getNumStreams() + 1

   for i = 1, self.inputTensorList2:len() do
      -- Assert consistency of tensor dimensions
      errorIfNot(#self.inputTensorList2[i].tensor:size() == #input:size(),
             "Tensor size mismatch: " ..
                #self.inputTensorList2[i].tensor:size() ..
                " vs " .. #input:size())
      errorIfNot(
         #self.gradOutputTensorList2[i].tensor:size() == #gradOutput:size())

      -- Set padding for this tile which can be partial and on the boundary
      local savePadUp, savePadLeft, savePadDown, savePadRight =
         self:pushPadding(i, self.inputTensorList2)

      local firstWrite = (i == 1)
      local lastWrite = (i == self.inputTensorList2:len())
      -- Interestingly, tiled input is reusable but has a long liveness
      -- If we don't reuse it we can reclaim the memory for something else
      -- This is all controlled by the bufferKey
      -- local reuseList = List{}
      local reuseList = List{}
      if self.inputTensorList and -- not cleaned earlier -> may want to reuse
         equalsTiledTensorDescriptor(self.inputTensorList[i],
                                     self.inputTensorList2[i]) then
         reuseList:append(self.FFTInputBufferType)
      end

      local inputLocal = self.inputTensorList2[i].tensor
      local outputLocal = self.gradOutputTensorList2[i].tensor
      local metaData = self.metaDataListAccGrad[i]
      local cublasBatchDims = 2
      -- 2D convolutions on 4D tensors atm
      assert(#inputLocal:size() == cublasBatchDims + 2)

      local commonSize = self:prepareSizeAndBuffers(
         inputLocal, self.gradWeight, outputLocal, metaData)

      -- Run all under this currentStream
      cutorch.setStream(currentStream)
      if not reuseList or not reuseList:contains(self.FFTOutputBufferType)
      then
         -- Potentially reuse buffer if so told
         local fftWrapper = nn.FFTWrapper(self.fftImplementation)
         fftWrapper:fft(outputLocal, self.outputBuffer, cublasBatchDims)
      else
         error('UpdateGradInput and AccGradParameter tiled padded ' ..
                  'gradOuput cannot be shared atm')
      end
      if not reuseList or not reuseList:contains(self.FFTInputBufferType)
      then
         -- Potentially reuse buffer if so told
         local fftWrapperPadded = nn.FFTWrapper(
            self.fftImplementation, self.padLeft, self.padUp)
         fftWrapperPadded:fft(
            inputLocal, self.inputBuffer, cublasBatchDims)
      end

      -- Because we accumulate into C, we must synchronize with the
      -- previous transposeMMFFI call. We statically know by construction
      -- that it leaves on previousStream and by transitivity of
      -- dependences we're good to go
      cutorch.streamWaitFor(currentStream, {previousStream})
      local lastWriteNorm = 1.0
      if lastWrite then
         local norm = self:getNormalizationFactor(commonSize, outputLocal)
         lastWriteNorm = (1.0 * scale) / norm
      end
      FBMMFFI.transposeMMFFI(cutorch._state,
                             self.outputBuffer:cdata(),
                             self.inputBuffer:cdata(),
                             self.weightBuffer:cdata(),
                             lastWriteNorm,
                             true,            -- conjugate A
                             false,           -- B
                                not firstWrite)  -- accumulate into C

      -- 3. Accumulate in the frequency domain, IFFT on last write
      if lastWrite then
         local fftWrapper = nn.FFTWrapper(self.fftImplementation)
         fftWrapper:ffti(
            self.gradWeight, self.weightBuffer, cublasBatchDims)
      end

      if self.printDebugLevel >= 3 then
         print('Step ASYNC gradWeight: ', self.gradWeight)
      end
      previousStream, currentStream =
         currentStream, currentStream % cutorch.getNumStreams() + 1

      -- Pop back saved padding values
      self.padUp, self.padLeft, self.padDown, self.padRight =
         savePadUp, savePadLeft, savePadDown, savePadRight
   end

   -- ##############################################
   cutorch.streamBarrier(self.allStreams)
end

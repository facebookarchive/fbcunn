-- Copyright 2004-present Facebook. All Rights Reserved.

require 'cudnn'
local List = require 'pl.List'
local ffi = require 'ffi'

local lib_name = 'torch_fb_fbcunn_convolution_bias'
local lib_path = package.searchpath(lib_name, package.cpath)
local ConvolutionBiasFFI = ffi.load(lib_path and lib_path or lib_name)

local lib_name = 'torch_fb_fbcunn_FFTIteratedConvolution'
local lib_path = package.searchpath(lib_name, package.cpath)
local FFTIteratedConvolution = ffi.load(lib_path and lib_path or lib_name)

------------------------------------------------------------------------------
--   Actual Module
------------------------------------------------------------------------------
local SpatialConvolutionFFTTiledIterated, parent =
   torch.class('nn.SpatialConvolutionFFTTiledIterated',
               'nn.SpatialConvolutionFFTTiled')

function SpatialConvolutionFFTTiledIterated:__init(nInputPlane,
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

   -- Override any memory reuse scheme: just no reuse
   self.memoryReusePolicy = List{nn.SpatialConvolutionFFT.memoryReuseNone}
end

-- Adjustment needed for updateGradInput since we don't do circular
-- shifts in the Fourier domain, just shift in time.
local function buildTiledDeviceTensorFFI(
      inputTensorList, outputTensorList, adjustInputShiftW, adjustInputShiftH)
   local adjustInputShiftW = adjustInputShiftW or 0
   local adjustInputShiftH = adjustInputShiftH or 0
   local size = inputTensorList:len()
   assert(outputTensorList:len() == size)
   local inputTiledDeviceTensorFFI =
      ffi.new("TiledDeviceTensorFFI[?]", size)
   local outputTiledDeviceTensorFFI =
      ffi.new("TiledDeviceTensorFFI[?]", size)
   for i = 1, size do
      inputTiledDeviceTensorFFI[i - 1].tensor =
         inputTensorList[i].tensor:cdata()
      inputTiledDeviceTensorFFI[i - 1].padL =
         inputTensorList[i].padLeft + adjustInputShiftW
      inputTiledDeviceTensorFFI[i - 1].padU =
         inputTensorList[i].padUp + adjustInputShiftH
      outputTiledDeviceTensorFFI[i - 1].tensor =
         outputTensorList[i].tensor:cdata()
      outputTiledDeviceTensorFFI[i - 1].padL = outputTensorList[i].padLeft
      outputTiledDeviceTensorFFI[i - 1].padU = outputTensorList[i].padUp
   end
   return inputTiledDeviceTensorFFI, outputTiledDeviceTensorFFI, size
end

function SpatialConvolutionFFTTiledIterated:instUpdateOutputFFTImpl(input)
   -- Make sure tiling information has been precomputed
   assert(self.inputTensorList)
   assert(self.outputTensorList)
   assert(self.memoryReusePolicy:contains(
             nn.SpatialConvolutionFFT.memoryReuseNone))

   local inputTiledDeviceTensorFFI, outputTiledDeviceTensorFFI, numTiles =
      buildTiledDeviceTensorFFI(self.inputTensorList, self.outputTensorList)


   for _, actualTileSize in ipairs({8, 16, 32}) do
      if self.memoryReusePolicy:contains(
         nn.SpatialConvolutionFFT.memoryReuseNone) and
         self.tileSizeH <= actualTileSize and self.tileSizeW <= actualTileSize
      then
         -- Only do iterated convolutions if there is no reuse
         self.output:zero()
         -- ##############################################
         cutorch.streamBarrier(self.allStreams)
         local convolutionPassFFI =
            ffi.new("FFTConvolutionPassFFI")
         convolutionPassFFI.pass = convolutionPassFFI.FFT_UpdateOutput

         FFTIteratedConvolution.convolveIteratedFFI(
            cutorch._state,
            inputTiledDeviceTensorFFI,
            self.weight:cdata(),
            outputTiledDeviceTensorFFI,
            numTiles,
            actualTileSize,
            convolutionPassFFI,
            1.0)

         -- ##############################################
         cutorch.streamBarrier(self.allStreams)
         ConvolutionBiasFFI.updateOutputBiasFFI(
            cutorch._state, self.output:cdata(), self.bias:cdata())
         -- ##############################################
         cutorch.streamBarrier(self.allStreams)
         return self.output
      end
   end

   error('updateOutputIterated tiling by ' .. self.tileSizeW .. 'x' ..
            self.tileSizeH .. ' not supported')
end



function SpatialConvolutionFFTTiledIterated:instUpdateGradInputFFTImpl(
      input, gradOutput)
   -- Make sure tiling information has been precomputed
   assert(self.gradInputTensorList)
   assert(self.gradOutputTensorList)
   assert(self.memoryReusePolicy:contains(
             nn.SpatialConvolutionFFT.memoryReuseNone))

   local gradInputTiledDeviceTensorFFI,
         gradOutputTiledDeviceTensorFFI,
         numTiles =
            buildTiledDeviceTensorFFI(self.gradInputTensorList,
                                      self.gradOutputTensorList,
                                      -- Adjust for no circular rotation in
                                      -- Fourier domain
                                      self.kW - 1,
                                      self.kH - 1
            )

   for _, actualTileSize in ipairs({8, 16, 32}) do
      if self.memoryReusePolicy:contains(
         nn.SpatialConvolutionFFT.memoryReuseNone) and
         self.tileSizeH <= actualTileSize and self.tileSizeW <= actualTileSize
      then
         -- Only do iterated convolutions if there is not reuse
         self.gradInput:zero()
         -- ##############################################
         cutorch.streamBarrier(self.allStreams)
         local convolutionPassFFI =
            ffi.new("FFTConvolutionPassFFI")
         convolutionPassFFI.pass = convolutionPassFFI.FFT_UpdateGradInput
         FFTIteratedConvolution.convolveIteratedFFI(
            cutorch._state,
            gradInputTiledDeviceTensorFFI,
            self.weight:cdata(),
            gradOutputTiledDeviceTensorFFI,
            numTiles,
            actualTileSize,
            convolutionPassFFI,
            1.0)
         -- ##############################################
         cutorch.streamBarrier(self.allStreams)
         return self.gradInput
      end
   end

   error('updateGradInputIterated tiling by ' .. self.tileSizeW .. 'x' ..
            self.tileSizeH .. ' not supported')
end


function SpatialConvolutionFFTTiledIterated:instAccGradParametersFFTImpl(
      input, gradOutput, scale)

   -- Make sure tiling information has been precomputed
   assert(self.inputTensorList2)
   assert(self.gradOutputTensorList2)
   assert(self.memoryReusePolicy:contains(
             nn.SpatialConvolutionFFT.memoryReuseNone))

   local inputTiledDeviceTensorFFI,
         gradOutputTiledDeviceTensorFFI,
         numTiles =
            buildTiledDeviceTensorFFI(self.inputTensorList2,
                                      self.gradOutputTensorList2)

   for _, actualTileSize in ipairs({8, 16, 32}) do
      if self.memoryReusePolicy:contains(
         nn.SpatialConvolutionFFT.memoryReuseNone) and
         self.tileSizeH <= actualTileSize and self.tileSizeW <= actualTileSize
      then
         -- Only do iterated convolutions if there is no reuse
         self.gradWeight:zero()
         -- ##############################################
         cutorch.streamBarrier(self.allStreams)

         -- Run ahead
         cutorch.setStream(1)
         ConvolutionBiasFFI.accGradParametersBiasFFI(
            cutorch._state, gradOutput:cdata(), self.gradBias:cdata(), scale)

         cutorch.setStream(2)
         local convolutionPassFFI =
            ffi.new("FFTConvolutionPassFFI")
         convolutionPassFFI.pass = convolutionPassFFI.FFT_AccGradParameters
         FFTIteratedConvolution.convolveIteratedFFI(
            cutorch._state,
            inputTiledDeviceTensorFFI,
            self.gradWeight:cdata(),
            gradOutputTiledDeviceTensorFFI,
            numTiles,
            actualTileSize,
            convolutionPassFFI,
            scale)
         -- ##############################################
         cutorch.streamBarrier(self.allStreams)
         return
      end
   end

   error('accGradParametersIterated tiling by ' .. self.tileSizeW .. 'x' ..
            self.tileSizeH .. ' not supported')
end

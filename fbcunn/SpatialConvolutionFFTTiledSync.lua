-- Copyright 2004-present Facebook. All Rights Reserved.

require 'cudnn'
local List = require 'pl.List'
local ffi = require 'ffi'
local ConvolutionBiasFFI = ffi.load('torch_fb_fbcunn_convolution_bias')

local function errorIf(cond, msg)
   if cond then
      error(msg)
   end
end

local function errorIfNot(cond, msg)
   errorIf(not cond, msg)
end

------------------------------------------------------------------------------
--   Actual Module
------------------------------------------------------------------------------
local SpatialConvolutionFFTTiledSync, parent =
   torch.class('nn.SpatialConvolutionFFTTiledSync',
               'nn.SpatialConvolutionFFTTiled')

function SpatialConvolutionFFTTiledSync:__init(nInputPlane,
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

function SpatialConvolutionFFTTiledSync:instUpdateOutputFFTImpl(input)
   -- Make sure tiling information has been precomputed
   assert(self.inputTensorList)
   assert(self.outputTensorList)
   assert(self.memoryReusePolicy:contains(
             nn.SpatialConvolutionFFT.memoryReuseNone))

   -- Push / pop the local tensor, we're calling a parent in sync mode
   local saveOutput = self.output
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

      -- Even in the absence of reuse we can compute the weight buffers only
      -- once. This is one of the points of tiling in the first place
      local firstIteration = (i == 1)
      local reuseList = List{}
      if not firstIteration then
         reuseList:append(self.FFTWeightBufferType)
      end
      self.output = self.outputTensorList[i].tensor
      -- Go up 2 levels, 'cast' as SpatialConvolutionFBFFT
      nn.SpatialConvolutionFBFFT.updateOutputFFTImpl(
         self,
         self.inputTensorList[i].tensor,
         reuseList,
         self.metaDataListUpdateOutput[i])

      -- Pop back saved padding values
      self.padUp, self.padLeft, self.padDown, self.padRight =
         savePadUp, savePadLeft, savePadDown, savePadRight
   end

   self.output = saveOutput
   -- ##############################################
   cutorch.streamBarrier(self.allStreams)
   cutorch.setStream(1)
   ConvolutionBiasFFI.updateOutputBiasFFI(
      cutorch._state, self.output:cdata(), self.bias:cdata())
   -- ##############################################
   cutorch.streamBarrier(self.allStreams)

   return self.output
end



function SpatialConvolutionFFTTiledSync:instUpdateGradInputFFTImpl(
      input, gradOutput)
   -- Make sure tiling information has been precomputed
   assert(self.gradInputTensorList)
   assert(self.gradOutputTensorList)
   assert(self.memoryReusePolicy:contains(
             nn.SpatialConvolutionFFT.memoryReuseNone))

   -- Push / pop the local tensor, we're calling a parent in sync mode
   local saveGradInput = self.gradInput
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
         reuseList:append(self.FFTWeightBufferType)
      end

      self.gradInput = self.gradInputTensorList[i].tensor
      -- Go up 2 levels, 'cast' as SpatialConvolutionFBFFT
      nn.SpatialConvolutionFBFFT.updateGradInputFFTImpl(
         self,
         self.gradInput, -- used only as model
         self.gradOutputTensorList[i].tensor,
         -- weight buffers can always be reused
         -- since we enforce that tiles are larger
         -- than weights
         reuseList,
         self.metaDataListUpdateGradInput[i])

      -- Pop back saved padding values
      self.padUp, self.padLeft, self.padDown, self.padRight =
         savePadUp, savePadLeft, savePadDown, savePadRight
   end
   self.gradInput = saveGradInput
   return self.gradInput
end


function SpatialConvolutionFFTTiledSync:instAccGradParametersFFTImpl(
      input, gradOutput, scale)
   -- Make sure tiling information has been precomputed
   assert(self.inputTensorList2)
   assert(self.gradOutputTensorList2)
   assert(self.memoryReusePolicy:contains(
             nn.SpatialConvolutionFFT.memoryReuseNone))

   -- ##############################################
   cutorch.streamBarrier(self.allStreams)

   -- Run ahead
   local currentStream = 0
   cutorch.setStream(currentStream)
   ConvolutionBiasFFI.accGradParametersBiasFFI(
      cutorch._state, gradOutput:cdata(), self.gradBias:cdata(), scale)

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

      -- We accumulate in this thing, make sure it is zero
      self.gradWeight:zero()
      self.gradBias:zero()

      if firstWrite then
         self.gradWeightAcc = self.gradWeight:clone()
         self.gradBiasAcc = self.gradBias:clone()
      end

      -- Can't reuse tiled gradOutput without extra work
      errorIf(self.memoryReusePolicy:contains(
                 nn.SpatialConvolutionFFT.memoryReuseOutput),
              "Reuse output in tiled accGradParameters is not supproted")

      if self.printDebugLevel >= 3 then
         print('Pre step synchronous gradWeight @',
               self.gradWeight:cdata(), ': ', self.gradWeight:float())
      end

      -- Go up 2 levels, 'cast' as SpatialConvolutionFBFFT
      nn.SpatialConvolutionFBFFT.accGradParametersFFTImpl(
         self,
         self.inputTensorList2[i].tensor,
         self.gradOutputTensorList2[i].tensor,
         scale,
         List{}, -- reuseList
         self.metaDataListAccGrad[i])

      -- Super heavy, need to clear this up
      -- ##############################################
      cutorch.streamBarrier(self.allStreams)
      self.gradWeightAcc:add(self.gradWeight)
      self.gradBiasAcc:add(self.gradBias)
      -- ##############################################
      cutorch.streamBarrier(self.allStreams)

      if self.printDebugLevel >= 3 then
         print('Step synchronous gradWeight @',
               self.gradWeight:cdata(), ': ', self.gradWeight:float())
      end

      if lastWrite then
         self.gradWeight:copy(self.gradWeightAcc)
         self.gradBias:copy(self.gradBiasAcc)
      end

      -- Pop back saved padding values
      self.padUp, self.padLeft, self.padDown, self.padRight =
         savePadUp, savePadLeft, savePadDown, savePadRight
   end

   -- ##############################################
   cutorch.streamBarrier(self.allStreams)
end

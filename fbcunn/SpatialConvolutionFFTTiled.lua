-- Copyright 2004-present Facebook. All Rights Reserved.

require 'cudnn'
local List = require 'pl.List'
local thrift = require('fb.thrift')

local function errorIf(cond, msg)
   if cond then
      error(msg)
   end
end

local function errorIfNot(cond, msg)
   errorIf(not cond, msg)
end

--[[
   Move to Tensor.lua

   This helper funtion returns a pl.List of 2-D tiled views into the tensor
   passed in input, corresponding to tiling by the specified tiles sizes, with
   specfied step sizes and implicit padding sizes.
   Tiling is performed on the innermost 2 dimensions so tensor:nDimension must
   be >= 2.

   -- TileDescriptor "declaration"
   local TiledTensorDescriptor = {}
   -- Original tile sizes asked for for proper Fourier basis decomposition
   TiledTensorDescriptor.tileSizeH = tileSizeH
   TiledTensorDescriptor.tileSizeW = tileSizeW
   -- Index of the tile in tile space
   TiledTensorDescriptor.tileIndexH = tileIndexH
   TiledTensorDescriptor.tileIndexW = tileIndexW
   -- Actual tensor size, full tiles have tensorSize == tileSize
   TiledTensorDescriptor.tensorSizeH = tensorSizeH
   TiledTensorDescriptor.tensorSizeW = tensorSizeW
   -- Up and Left padding for up and left boundary tile.
   -- Down and Right are obtained by implicit zero padding up to
   -- original tile size
   TiledTensorDescriptor.padUp = padUp
   TiledTensorDescriptor.padLeft = padLeft
   -- The view in the original tensor
   TiledTensorDescriptor.tensor = torch.Tensor()

   By default tiling returns all the subtensors, including partial tensors on
   the boundaries, that have at least one element when traversed by
   tileSizeH x tileSizeW with stride stepH x stepW.
   When performing convolutions, tiling semantics may not be sufficient.
   For consistency, the tiling of the tensor written into, informs how many
   tiles we should obtain from the tensor read from; this information is
   conveyed by numTilesH x numTilesW.
   The consistency check is that the tiling of the tensor read from, must always
   cover the full read tensor.
--]]
local function TiledView2D(tensor,
                           tileSizeH,
                           tileSizeW,
                           stepH,
                           stepW,
                           padLeft,
                           padUp,
                           padRight,
                           padDown,
                           numTilesH,
                           numTilesW)
   -- Initialization
   local stepH = stepH or tileSizeH
   local stepW = stepW or tileSizeW
   local padUp = padUp or 0
   local padLeft = padLeft or 0
   local padDown = padDown or 0
   local padRight = padRight or 0
   local dimIndexH = tensor:nDimension() - 1
   local dimIndexW = tensor:nDimension()
   local numTilesH = numTilesH or 1e100 -- maxint would be nice
   local numTilesW = numTilesW or 1e100 -- maxint would be nice

   local printDebugLevel = -1
   if printDebugLevel >= 1 then
      print("Tile ", tensor:size(), " by ", tileSizeH, "x", tileSizeW,
            " with step ", stepH, "x", stepW, " and pad ",
            padUp, "x", padLeft, "x", padDown, "x", padRight)
   end

   -- Input validation, reject padding larger than tile size or kernel size
   assert(tensor:nDimension() >= 2)
   assert(tileSizeH and tileSizeW, 'both tile sizes must be specified')
   assert(padUp >= 0 and padUp < tileSizeH, "padUp = " .. padUp ..
             " >= (incompatible with) tileSizeH = " .. tileSizeH)
   assert(padLeft >= 0 and padLeft < tileSizeW, "padLeft = " .. padLeft ..
             " >= (incompatible with) with tileSizeW = " .. tileSizeW)
   assert(padDown >= 0 and padDown < tileSizeH, "padDown = " .. padDown ..
             " >= (incompatible with) with tileSizeH = " .. tileSizeH)
   assert(padRight >= 0 and padRight < tileSizeW, "padRight = " .. padRight ..
             " >= (incompatible with) with tileSizeW = " .. tileSizeW)
   assert(tileSizeW > 0 and tileSizeH > 0, "")
   assert(stepH > 0 and stepW > 0,
          "Step sizes " .. stepH .. " x " .. stepW .. " both expected > 1. " ..
             "Otherwise, tileSize <= kernel size which should not occur")
   assert(padUp >= 0 and padDown >= 0 and padLeft >= 0 and padRight >= 0)
   errorIfNot(tileSizeH < tensor:size(dimIndexH),
              "Tiling must be smaller than tensor size !")
   errorIfNot(tileSizeW < tensor:size(dimIndexW),
              "Tiling must be smaller than tensor size !")
   assert(#tensor:size() == dimIndexW and #tensor:stride() == dimIndexW)

   -- TileDescriptor generating loop
   local maxTileIndexH = 0
   local maxTileIndexW = 0
   local tensors = List{}
   local tensorStrideH = tensor:stride(dimIndexH)
   local tensorStrideW = tensor:stride(dimIndexW)
   local tileIndexH = 0
   for y = -padUp + 1, tensor:size(dimIndexH), stepH do

      -- Continue would be nice here to avoid level of nesting !
      if tileIndexH < numTilesH then

         local tileIndexW = 0
         for x = -padLeft + 1, tensor:size(dimIndexW), stepW do

            -- Continue would be nice here to avoid level of nesting !
            if tileIndexW < numTilesW then

               -- Descriptor for each tiled tensor
               local TiledTensorDescriptor = {}

               -- Handle special boundary case for partial tile along y
               local tensorSizeH = 0
               if y <= 0 then
                  tensorSizeH = tileSizeH + (y-1)
                  TiledTensorDescriptor.padUp = -(y-1) -- padUp
               else
                  -- If we generate a tile, make sure its size does not overflow
                  tensorSizeH = math.max(
                     1, math.min(tileSizeH, tensor:size(dimIndexH) - (y-1)))
                  TiledTensorDescriptor.padUp = 0
               end
               TiledTensorDescriptor.tensorSizeH = tensorSizeH
               TiledTensorDescriptor.tileIndexH = tileIndexH

               -- Handle special boundary case for partial tile along x
               local tensorSizeW = 0
               if x <= 0 then
                  tensorSizeW = tileSizeW + (x-1)
                  TiledTensorDescriptor.padLeft = -(x-1) -- padLeft
               else
                  -- If we generate a tile, make sure its size does not overflow
                  tensorSizeW = math.max(
                     1, math.min(tileSizeW, tensor:size(dimIndexW) - (x-1)))
                  TiledTensorDescriptor.padLeft = 0
               end
               TiledTensorDescriptor.tensorSizeW = tensorSizeW
               TiledTensorDescriptor.tileIndexW = tileIndexW

               -- Allocate tensor with partial or full size and full stride
               -- for proper wraparound
               local sizes =
                  torch.LongStorage(tensor:nDimension()):copy(tensor:size())
               sizes[#sizes - 1] = tensorSizeH
               sizes[#sizes] = tensorSizeW
               local tensorTiled = torch.Tensor():typeAs(tensor)
               tensorTiled:set(
                  tensor:storage(),
                  tensor:storageOffset() +
                     math.max((y-1), 0) * tensorStrideH +
                     math.max((x-1), 0) * tensorStrideW,
                  sizes,
                  tensor:stride())

               TiledTensorDescriptor.tileSizeH = tileSizeH
               TiledTensorDescriptor.tileSizeW = tileSizeW
               TiledTensorDescriptor.tensor = tensorTiled

               -- Handling partial til on the bottom and right sides
               -- Important to get interpolation right in frequency domain
               tensors:append(TiledTensorDescriptor)

               if printDebugLevel >= 1 then
                  print('y = ' .. y .. ' x = ' .. x ..
                           ' tile index = ' .. tileIndexH .. ' x '.. tileIndexW)
                  print(TiledTensorDescriptor)
                  if printDebugLevel >= 2 then
                     print(TiledTensorDescriptor.tensor)
                  end
               end

               assert(tensor:size(dimIndexH) + padUp + padDown -
                         tileIndexH * stepH > 0, "Error tileIndexH = " ..
                         tileIndexH .. " stepH = " .. stepH)
               assert(tensor:size(dimIndexW) + padLeft + padRight -
                         tileIndexW * stepW > 0, "Error tileIndexW = " ..
                         tileIndexW .. " stepW = " .. stepW)
               assert(tensorSizeH > 0, 'tensorSizeH = ' .. tensorSizeH)
               assert(tensorSizeW > 0, 'tensorSizeW = ' .. tensorSizeW)
               assert(y <= tensor:size(dimIndexH), 'Overflow y = ' .. y ..
                         ' > size = ' .. tensor:size(dimIndexH))
               assert(x <= tensor:size(dimIndexW), 'Overflow x = ' .. x ..
                         ' > size = ' .. tensor:size(dimIndexW))


               if maxTileIndexW < tileIndexW then
                  maxTileIndexW = tileIndexW
               end
               tileIndexW = tileIndexW + 1
            else  -- if tileIndexW < numTilesW
               assert(x + tileSizeW - stepW >= tensor:size(dimIndexW))
            end -- if tileIndexW < numTilesW
         end -- for x

         if maxTileIndexH < tileIndexH then
            maxTileIndexH = tileIndexH
         end
         tileIndexH = tileIndexH + 1
      else -- if not tileIndexH < numTilesH
         assert(y + tileSizeH - stepH >= tensor:size(dimIndexH))
      end -- if tileIndexH < numTilesH
   end -- for y

   return tensors, maxTileIndexH, maxTileIndexW
end

-- Not really a string but I want to print this structure
local function TiledTensorDescriptorToString(TiledTensorDescriptor)
   local toPrint = {}
   toPrint.td = TiledTensorDescriptor
   toPrint.tensorAddress = TiledTensorDescriptor.tensor:cdata()
   toPrint.storageAddress = TiledTensorDescriptor.tensor:storage():cdata()
   toPrint.storageOffset = TiledTensorDescriptor.tensor:storageOffset()
   return toPrint
end

local function _printDebugAndAssert(
      debugLevel, index, inputTensorList, outputTensorList)
   if debugLevel == 1 then
      print("Convolve input", index, " / ",
            outputTensorList:len(), " :\n",
            TiledTensorDescriptorToString(inputTensorList[index]),
            '\n Convolve output\n',
            TiledTensorDescriptorToString(outputTensorList[index]))
   elseif debugLevel >= 2 then
      print("Convolve input", index, " / ",
            outputTensorList:len(), " :\n",
            TiledTensorDescriptorToString(inputTensorList[index]),
            inputTensorList[index].tensor,
            '\n Convolve output\n',
            TiledTensorDescriptorToString(outputTensorList[index]),
            outputTensorList[index].tensor)
   end

   -- Assert tiles are traversed in the same order otherwise
   -- you can forget about correctness
   assert(outputTensorList[index].tileIndexH ==
             inputTensorList[index].tileIndexH)
   assert(outputTensorList[index].tileIndexW ==
             inputTensorList[index].tileIndexW)
end

------------------------------------------------------------------------------
--   Actual Module
------------------------------------------------------------------------------
local SpatialConvolutionFFTTiled, parent =
   torch.class('nn.SpatialConvolutionFFTTiled', 'nn.SpatialConvolutionFBFFT')

function SpatialConvolutionFFTTiled:__init(nInputPlane,
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

   assert(torch.type(nInputPlane) == 'number')
   assert(torch.type(nOutputPlane) == 'number')
   assert(torch.type(kW) == 'number')
   assert(torch.type(kH) == 'number')
   assert(torch.type(dW) == 'number')
   assert(torch.type(dH) == 'number')
   assert(padLeft == nil or torch.type(padLeft) == 'number')
   assert(padUp == nil or torch.type(padUp) == 'number')

   assert(tileSizeW == nil or torch.type(tileSizeW) == 'number')
   assert(tileSizeH == nil or torch.type(tileSizeH) == 'number')
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

   -- Sanity assertions
   assert(self.printDebugLevel == -1)
   assert(self.nInputPlane == nInputPlane)
   assert(self.nOutputPlane == nOutputPlane)
   assert(self.kW == kW)
   assert(self.kH == kH)
   assert(self.dH == 1, "fft only supports stride-1 convolutions atm")
   assert(self.dW == 1, "fft only supports stride-1 convolutions atm")

   assert(self.padLeft == padLeft or self.padLeft == 0)
   assert(self.padUp == padUp or self.padUp == 0)
   assert(self.padRight == self.padLeft)
   assert(self.padDown == self.padUp)

   assert(self.fftImplementation == 'fbfft')

   assert(self.padUp < self.kH and self.padDown < self.kH and
             self.padLeft < self.kW and self.padRight < self.kW,
          "Padding must be smaller than kernel")

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


   -- Temporary buffers, would be nice to reduce code size here
   assert(not self.inputBuffer)
   assert(not self.inputTransposeBuffer)
   assert(not self.inputPadded)
   assert(not self.outputBuffer)
   assert(not self.outputTransposeBuffer)
   assert(not self.outputPadded)
   assert(not self.weightBuffer)
   assert(not self.weightTransposeBuffer)
   assert(not self.weightPadded)

   -- CuFFT plans, useless for fbfft
   assert(not self.cufftPlanInputFFT)
   assert(not self.cufftPlanWeightFFT)
   assert(not self.cufftPlanOutputFFT)
   assert(not self.cufftPlanInputIFFT)
   assert(not self.cufftPlanWeightIFFT)
   assert(not self.cufftPlanOutputIFFT)

   self:reset()

   -- Tiling metadata
   self.tileSizeH = tileSizeH or 16
   self.tileSizeW = tileSizeW or 16
   -- updateOutput
   self.inputTensorList = nil
   self.outputTensorList = nil
   -- updateGradInput
   self.gradInputTensorList = nil
   self.gradOutputTensorList = nil
   -- accGradParameters
   self.inputTensorList2 = nil
   self.gradOutputTensorList2 = nil
end


local function printDebugAndAssert(
      debugLevel, index, inputTensorList, outputTensorList)
   if debugLevel == 1 then
      print("Convolve input", index, " / ",
            outputTensorList:len(), " :\n",
            TiledTensorDescriptorToString(inputTensorList[index]),
            '\n Convolve output\n',
            TiledTensorDescriptorToString(outputTensorList[index]))
   elseif debugLevel >= 2 then
      print("Convolve input", index, " / ",
            outputTensorList:len(), " :\n",
            TiledTensorDescriptorToString(inputTensorList[index]),
            inputTensorList[index].tensor,
            '\n Convolve output\n',
            TiledTensorDescriptorToString(outputTensorList[index]),
            outputTensorList[index].tensor)
   end

   -- Assert tiles are traversed in the same order otherwise
   -- you can forget about correctness
   assert(outputTensorList[index].tileIndexH ==
             inputTensorList[index].tileIndexH)
   assert(outputTensorList[index].tileIndexW ==
             inputTensorList[index].tileIndexW)
end


function SpatialConvolutionFFTTiled:pushPadding(index, tensorList)
   local savePadUp, savePadLeft, savePadDown, savePadRight
   savePadUp, self.padUp = self.padUp, tensorList[index].padUp
   savePadLeft, self.padLeft = self.padLeft, tensorList[index].padLeft
   -- Complete padding up to tile size so that interpolation
   -- occurs in the right Fourier basis
   savePadDown, self.padDown =
      self.padDown, math.max(
         0, tensorList[index].tileSizeH -
            (self.padUp + tensorList[index].tensorSizeH))
   savePadRight, self.padRight =
      self.padRight, math.max(
         0, tensorList[index].tileSizeW -
            (self.padLeft + tensorList[index].tensorSizeW))

   return savePadUp, savePadLeft, savePadDown, savePadRight
end


function SpatialConvolutionFFTTiled:pushPaddingWithCircularSymmetry(
      index)
   local savePadUp, savePadLeft, savePadDown, savePadRight
   -- Fun with padding and circular symmetry in Fourier domain
   -- This acts upon shifting the IFFT result into the proper position
   -- into gradInput
   savePadUp, self.padUp =
      self.padUp, self.kH - 1 + self.gradInputTensorList[index].padUp -
      self.gradOutputTensorList[index].padUp
   savePadLeft, self.padLeft =
      self.padLeft, self.kW - 1 + self.gradInputTensorList[index].padLeft -
      self.gradOutputTensorList[index].padLeft
   -- Complete padding up to tile size so that interpolation
   -- occurs in the right Fourier basis.
   -- The invariant is that the size of gradOutput and gradInput should
   -- always be padded up to the tiling size. In the particular case
   -- of gradInput, we must additionally consider input padding.

   assert(self.gradOutputTensorList[index].tensorSizeH)
   assert(self.gradInputTensorList[index].tensorSizeH)

   savePadDown, self.padDown =
      self.padDown,
   math.max(0, self.tileSizeH - math.max(
               self.gradOutputTensorList[index].tensorSizeH,
               self.gradInputTensorList[index].tensorSizeH + self.padUp))
   savePadRight, self.padRight =
      self.padRight,
   math.max(0, self.tileSizeW - math.max(
               self.gradOutputTensorList[index].tensorSizeW,
               self.gradInputTensorList[index].tensorSizeW + self.padLeft))
   return savePadUp, savePadLeft, savePadDown, savePadRight
end

function SpatialConvolutionFFTTiled:updateOutputFFTImpl(input)
   local ok, res =
      pcall(SpatialConvolutionFFTTiled.abstractUpdateOutputFFTImpl, self, input)
   if ok then
      return res
   end
   self.success = false
   if self.reportErrors then
      print(res .. " -> updateOutput fallback to untiled FBFFT")
   end

   -- This path exits early for tuned SpatialConvolution.lua
   self.success = false
   if self.autotuningPass then
      error('Using tuned SpatialConvolution and found an error, early exit')
   end

   error("Bug in fallback form Tiled to FBFFT on updateOutput" ..
            " Drop back higher up in the food chain")
   -- This path is becoming obsolete
   -- Safety barrier and no reuse for error recovery
   self.memoryReusePolicy = List{
      nn.SpatialConvolutionFFT.memoryReuseNone}
   -- ##############################################
   cutorch.streamBarrier(self.allStreams)
   return parent.updateOutputFFTImpl(self, input)
end


function SpatialConvolutionFFTTiled:instUpdateOutputFFTImpl(
      input, gradOutput)
   assert(false, "Do not call the abstract class directly!")
end


function SpatialConvolutionFFTTiled:abstractUpdateOutputFFTImpl(input)
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

   errorIf(self.tileSizeH > self.output:size(3) or
              self.tileSizeW > self.output:size(4),
           'Tile size too large (' .. self.tileSizeH .. 'x' .. self.tileSizeW ..
            ') for output (' .. self.output:size(3) .. 'x' ..
            self.output:size(4) .. ')')

   -- Perform tiling on meta-tensor list
   if not self.inputTensorList or
      not self.outputTensorList or
      not self.metaDataListUpdateOutput
   then
      self.inputTensorList = nil
      self.outputTensorList = nil
      self.metaDataListUpdateOutput = nil
      local maxTileIndexH
      local maxTileIndexW
      -- In updateOutputTiled, the tiling of output is without overlap
      -- and without padding. It informs how the tiling on padded input
      -- should be performed
      self.outputTensorList, maxTileIndexH, maxTileIndexW =
         TiledView2D(self.output,
                     self.tileSizeH - self.kH + 1,
                     self.tileSizeW - self.kW + 1,
                     self.tileSizeH - self.kH + 1,
                     self.tileSizeW - self.kW + 1)
      self.inputTensorList = TiledView2D(input,
                                         self.tileSizeH,
                                         self.tileSizeW,
                                         self.tileSizeH - self.kH + 1,
                                         self.tileSizeW - self.kW + 1,
                                         self.padLeft,
                                         self.padUp,
                                         self.padRight,
                                         self.padDown,
                                         maxTileIndexH + 1,
                                         maxTileIndexW + 1)

      self.metaDataListUpdateOutput = List{}
      for i = 1, self.inputTensorList:len() do
         local metaData = self:makeMetaData(
            nn.SpatialConvolutionFFT.ForwardFFTPass,
            self.inputTensorList[i].tileIndexW,
            self.inputTensorList[i].tileIndexH,
            self.outputTensorList[i].tileIndexW,
            self.outputTensorList[i].tileIndexH)
         -- By default skip bias when offloading computation to FBFFT
         -- and do it at the very end
         metaData.skipBias = true
         self.metaDataListUpdateOutput:append(metaData)
      end
   end

   errorIfNot(self.outputTensorList:len() == self.inputTensorList:len(),
              "Error in tile metadata: not the same sizes input = " ..
                 self.inputTensorList:len() .. " VS output = " ..
                 self.outputTensorList:len())

   -- At this point tiles / metadata for buffer management / reuse are available
   -- in self.xyz just call the actual instantiation

   return self:instUpdateOutputFFTImpl(input)
end


function SpatialConvolutionFFTTiled:updateGradInputFFTImpl(input, gradOutput)
   local ok, res =
      pcall(SpatialConvolutionFFTTiled.abstractUpdateGradInputFFTImpl,
            self,
            input,
            gradOutput)
   if ok then
      return res
   end
   self.success = false
   if self.reportErrors then
      print(res .. " -> updateGradInput fallback to untiled FBFFT")
   end

   -- This path exits early for tuned SpatialConvolution.lua
   self.success = false
   if self.autotuningPass then
      error('Using tuned SpatialConvolution and found an error, early exit')
   end

   error("Bug in fallback form Tiled to FBFFT on updateGradInput" ..
            " Drop back higher up in the food chain")
   -- Safety barrier and no reuse for error recovery
   self.memoryReusePolicy = List{
      nn.SpatialConvolutionFFT.memoryReuseNone}
   -- ##############################################
   cutorch.streamBarrier(self.allStreams)
   return parent.updateGradInputFFTImpl(self, input, gradOutput)
end

function SpatialConvolutionFFTTiled:instUpdateGradInputFFTImpl(
      input, gradOutput)
   assert(false, "Do not call the abstract class directly!")
end

function SpatialConvolutionFFTTiled:abstractUpdateGradInputFFTImpl(
      input, gradOutput)
   assert(torch.type(input) == 'torch.CudaTensor', "CUDA support only!")

   local nBatches = input:size(1)

   -- Allocate the gradInput for this module, only once
   if not self.gradInput or self.gradInput:nElement() == 0 then
      self.gradInput = torch.CudaTensor(torch.LongStorage({
         nBatches,
         self.nInputPlane,
         input:size(3),
         input:size(4)}))
   else
      errorIfNot(self.gradInput:size(1) == input:size(1))
      errorIfNot(self.gradInput:size(2) == input:size(2))
      errorIfNot(self.gradInput:size(3) == input:size(3))
      errorIfNot(self.gradInput:size(4) == input:size(4))
   end

   errorIf(self.tileSizeH > gradOutput:size(3) or
              self.tileSizeW > gradOutput:size(4),
           'Tile size too large (' .. self.tileSizeH .. 'x' .. self.tileSizeW ..
           ') for gradOutput (' .. gradOutput:size(3) .. 'x' ..
           gradOutput:size(4) .. ')')

   -- Perform tiling on meta-tensor list
   if not self.gradOutputTensorList or
      not self.gradInputTensorList or
      not self.metaDataListUpdateGradInput
   then
      self.gradOutputTensorList = nil
      self.gradInputTensorList = nil
      self.metaDataListUpdateGradInput = nil
      local maxTileIndexH
      local maxTileIndexW
      -- In updateGradInputTiled, the tiling of gradInput is without overlap
      -- and with padding. It informs how the tiling on padded gradOutput
      -- should be performed.
      self.gradInputTensorList, maxTileIndexH, maxTileIndexW =
         TiledView2D(self.gradInput,
                     self.tileSizeH - self.kH + 1,
                     self.tileSizeW - self.kW + 1,
                     self.tileSizeH - self.kH + 1,
                     self.tileSizeW - self.kW + 1,
                     self.padLeft,
                     self.padUp,
                     self.padRight,
                     self.padDown)
      self.gradOutputTensorList = TiledView2D(gradOutput,
                                              self.tileSizeH,
                                              self.tileSizeW,
                                              self.tileSizeH - self.kH + 1,
                                              self.tileSizeW - self.kW + 1,
                                              self.kW - 1,
                                              self.kH - 1,
                                              self.kW - 1,
                                              self.kH - 1,
                                              maxTileIndexH + 1,
                                              maxTileIndexW + 1)
      self.metaDataListUpdateGradInput = List{}
      for i = 1, self.gradInputTensorList:len() do
         local metaData = self:makeMetaData(
            nn.SpatialConvolutionFFT.BackwardFFTPass,
            self.gradInputTensorList[i].tileIndexW,
            self.gradInputTensorList[i].tileIndexH,
            self.gradOutputTensorList[i].tileIndexW,
            self.gradOutputTensorList[i].tileIndexH)
         self.metaDataListUpdateGradInput:append(metaData)
      end
   end

   errorIfNot(self.gradInputTensorList:len() == self.gradOutputTensorList:len(),
          "Not the same sizes input = " .. self.gradOutputTensorList:len() ..
             " VS output = " .. self.gradInputTensorList:len())


   -- At this point tiles / metadata for buffer management / reuse are available
   -- in self.xyz just call the actual instantiation

   return self:instUpdateGradInputFFTImpl(input, gradOutput)
end


function SpatialConvolutionFFTTiled:accGradParametersFFTImpl(
      input, gradOutput, scale)
   local ok, res =
      pcall(SpatialConvolutionFFTTiled.abstractAccGradParametersFFTImpl,
            self,
            input,
            gradOutput,
            scale)
   if ok then
      return res
   end
   self.success = false
   if self.reportErrors then
      print(res .. " -> accGradParameters fallback to untiled FBFFT")
   end

   -- This path exits early for tuned SpatialConvolution.lua
   self.success = false
   if self.autotuningPass then
      error('Using tuned SpatialConvolution and found an error, early exit')
   end

   error("Bug in fallback form Tiled to FBFFT on accGradParametersFFTImpl" ..
            " Drop back higher up in the food chain")
   -- Safety barrier and no reuse for error recovery
   self.memoryReusePolicy = List{
      nn.SpatialConvolutionFFT.memoryReuseNone}
   -- ##############################################
   cutorch.streamBarrier(self.allStreams)
   parent.accGradParametersFFTImpl(self, input, gradOutput, scale)
end


function SpatialConvolutionFFTTiled:instAccGradParametersFFTImpl(
      input, gradOutput)
   assert(false, "Do not call the abstract class directly!")
end


function SpatialConvolutionFFTTiled:abstractAccGradParametersFFTImpl(
      input, gradOutput, scale)
   assert(torch.type(input) == 'torch.CudaTensor', "CUDA support only!")

   local scale = scale or 1
   local nBatches = input:size(1)

   -- Allocate the gradWeight for this module, only once
   if not self.gradWeight or self.gradWeight:nElement() == 0 then
      errorIfNot(false,
                 "GradWeight must already be allocated at module creation")
      self.gradWeight = torch.CudaTensor(torch.LongStorage({
                                               nBatches,
                                               self.nInputPlane,
                                               self.kH,
                                               self.kW}))
   end

   errorIf(self.tileSizeH > gradOutput:size(3) or
              self.tileSizeW > gradOutput:size(4),
           'Tile size too large (' .. self.tileSizeH .. 'x' .. self.tileSizeW ..
            ') for gradOutput (' .. gradOutput:size(3) .. 'x' ..
            gradOutput:size(4) .. ')')

   -- Perform tiling on meta-tensor list
   if not self.gradOutputTensorList2 or
         not self.inputTensorList2 or
         not self.metaDataListAccGrad then
      self.gradOutputTensorList2 = nil
      self.inputTensorList2 = nil
      self.metaDataListAccGrad = nil
      local maxTileIndexH
      local maxTileIndexW
      errorIfNot(self.tileSizeH >= self.kH,
                 'Tiling cannot be smaller than kernel !')
      errorIfNot(self.tileSizeW >= self.kW,
                 'Tiling cannot be smaller than kernel !')
      -- In updateGradInputTiled, the tiling of gradOutput is without overlap
      -- and without padding. It informs how the tiling on padded input
      -- should be performed.
      self.gradOutputTensorList2, maxTileIndexH, maxTileIndexW =
         TiledView2D(gradOutput,
                     self.tileSizeH - (self.kH - 1),
                     self.tileSizeW - (self.kW - 1),
                     self.tileSizeH - (self.kH - 1),
                     self.tileSizeW - (self.kW - 1))
      self.inputTensorList2 = TiledView2D(input,
                                          self.tileSizeH,
                                          self.tileSizeW,
                                          self.tileSizeH - (self.kH - 1),
                                          self.tileSizeW - (self.kW - 1),
                                          self.padLeft,
                                          self.padUp,
                                          self.padRight,
                                          self.padDown,
                                          maxTileIndexH + 1,
                                          maxTileIndexW + 1)

      self.metaDataListAccGrad = List{}
      for i = 1, self.inputTensorList2:len() do
         local metaData = self:makeMetaData(
            nn.SpatialConvolutionFFT.AccGradientFFTPass,
            self.inputTensorList2[i].tileIndexW,
            self.inputTensorList2[i].tileIndexH,
            self.gradOutputTensorList2[i].tileIndexW,
            self.gradOutputTensorList2[i].tileIndexH)
         self.metaDataListAccGrad:append(metaData)
      end
   end

   errorIfNot(self.inputTensorList2:len() == self.gradOutputTensorList2:len(),
          "Not the same sizes input = " .. self.gradOutputTensorList2:len() ..
             " VS output = " .. self.inputTensorList2:len())

   -- At this point tiles / metadata for buffer management / reuse are available

   self:instAccGradParametersFFTImpl(input, gradOutput, scale)
end

-- Makes or reuses square FFT buffers up to the next power of 2
function SpatialConvolutionFFTTiled:prepareSizeAndBuffers(i, w, o, metaData)
   return parent.prepareSizeAndBuffers(self, i, w, o, metaData)
end

function SpatialConvolutionFFTTiled:makeMetaData(
      pass,
      inputTileIndexW, inputTileIndexH,
      outputTileIndexW, outputTileIndexH,
      weightTileIndexW, weightTileIndexH)
   local metaData = {}
   metaData.pass = pass
   metaData.input = {}
   metaData.input.tileIndexH = inputTileIndexH
   metaData.input.tileIndexW = inputTileIndexW
   metaData.output = {}
   metaData.output.tileIndexH = outputTileIndexH
   metaData.output.tileIndexW = outputTileIndexW
   metaData.weight = {}
   metaData.weight.tileIndexH = weightTileIndexH
   metaData.weight.tileIndexW = weightTileIndexW
   return metaData
end

-- Discriminated buffers based on bufferType, bufferSize, tileIndex and
-- whether it is an input or an output "of the algorithm"
function SpatialConvolutionFFTTiled:getBufferKey(
      BufferType, bufferSizes, metaData)
   assert(torch.type(bufferSizes) == 'torch.LongStorage',
          torch.type(bufferSizes))
   assert(torch.type(metaData) == 'table',
          torch.type(metaData))

   -- TODO: needs semantics for proper producer consumer dependences and
   -- ordering for RAW dependences by using self.moduleTimeStep properly
   local md = {}
   if metaData then
      if BufferType == nn.SpatialConvolutionFFT.FFTInputBufferType then
         md.tileIndices = metaData.input
      elseif BufferType == nn.SpatialConvolutionFFT.FFTOutputBufferType then
         md.tileIndices = metaData.output
      else
         md.tileIndices = metaData.weight
      end

      -- This is an adhoc way to discriminate between
      --   updateOutput   / updateGradInput      / accGradParameters
      --   input  (false) /   gradInput  (true)  / input      (false)
      --   output (true)  /   gradOutput (false) / input      (false)
      --   weight (false) /   weight     (false) / gradWeight (true)
      --
      local isOutputOfAlgorithm = false
      -- In cufft mode, the tiled complex buffers are reused
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
      if torch.type(self) ~= "nn.SpatialConvolutionFFTTiledAsync" then
         -- if we run async we must have multiple tiles live at the same time,
         -- just let all tiles be live at the same time
         md = nil
      end
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

   if self.printDebugLevel >= 3 then
      print("BufferKey: ", bufferKey)
      print("Serialized to : ", res)
   end

   return res
end

function SpatialConvolutionFFTTiled:cleanupBuffers()
   parent.cleanupBuffers(self)

   -- Tiling metadata
   -- updateOutput
   self.inputTensorList = nil
   self.outputTensorList = nil
   self.metaDataListUpdateOutput = nil
   -- updateGradInput
   self.gradInputTensorList = nil
   self.gradOutputTensorList = nil
   self.metaDataListUpdateGradInput = nil
   -- accGradParameters
   self.inputTensorList2 = nil
   self.gradOutputTensorList2 = nil
   self.metaDataListAccGrad = nil

end

--[[
   This file implements Batch Normalization as described in the paper:
   "Batch Normalization: Accelerating Deep Network Training
                         by Reducing Internal Covariate Shift"
                   by Sergey Ioffe, Christian Szegedy

   This implementation is useful for inputs NOT coming from convolution layers.
   For Convolution layers, see SpatialBatchNormalization.lua

   The operation implemented is:
   y =     ( x - mean(x) )
        -------------------- * gamma + beta
       standard-deviation(x)
   where gamma and beta are learnable parameters.

   The learning of gamma and beta is optional.

   Usage:
   with    learnable parameters: nn.BatchNormalization(N [, eps] [,momentum])
                                 where N = dimensionality of input
   without learnable parameters: nn.BatchNormalization(0 [, eps] [,momentum])

   eps is a small value added to the standard-deviation to avoid divide-by-zero.
       Defaults to 1e-5

   Training: this layer keeps a running estimate of it's computed mean and std.
   The running sum is kept with a default momentup of 0.1 (unless over-ridden)
   Testing: this running mean/std is used to normalize.
]]--


local ffi = require 'ffi'

ffi.cdef[[
   void BatchNormalizationUpdateOutputFFI(
       THCState* state,
       THCudaTensor* input,
       THCudaTensor* output,
       THCudaTensor* centered,
       THCudaTensor* std,
       THCudaTensor* normalized,
       THCudaTensor* runningMean,
       THCudaTensor* runningStddev,
       THCudaTensor* weight,
       THCudaTensor* bias,
       float epsilon,
       float momentum,
       bool train,
       bool affine);
   void BatchNormalizationUpdateGradInputFFI(
       THCState* state,
       THCudaTensor* gradInput,
       THCudaTensor* gradOutput,
       THCudaTensor* centered,
       THCudaTensor* std,
       THCudaTensor* weight,
       bool affine);
   void BatchNormalizationAccGradParametersFFI(
       THCState* state,
       THCudaTensor* gradOutput,
       THCudaTensor* normalized,
       THCudaTensor* gradWeight,
       THCudaTensor* gradBias,
       float scale);
]]

local lib_name = 'torch_fb_fbcunn_batch_norm'
local lib_path = package.searchpath(lib_name, package.cpath)
local BNFFI = ffi.load(lib_path and lib_path or lib_name)

local BN, parent = torch.class('fbnn.BatchNormalization', 'nn.Module')

function BN:__init(nOutput, eps, momentum, affine)
   parent.__init(self)
   assert(nOutput and type(nOutput) == 'number',
          'Missing argument #1: dimensionality of input. ')
   assert(nOutput ~= 0, 'To set affine=false call BatchNormalization'
     .. '(nOutput,  eps, momentum, false) ')
   if affine ~= nil then
      assert(type(affine) == 'boolean', 'affine has to be true/false')
      self.affine = affine
   else
      self.affine = true
   end
   self.eps = eps or 1e-5
   self.train = true
   self.momentum = momentum or 0.1
   self.running_mean = torch.zeros(nOutput):cuda()
   self.running_std = torch.ones(nOutput):cuda()

   if self.affine then
      self.weight = torch.CudaTensor(nOutput)
      self.bias = torch.CudaTensor(nOutput)
      self.gradWeight = torch.CudaTensor(nOutput)
      self.gradBias = torch.CudaTensor(nOutput)
      self:reset()
   else
      -- Give me empty tensors for proper FFI behavior
      self.weight = torch.CudaTensor()
      self.bias = torch.CudaTensor()
      self.gradWeight = torch.CudaTensor()
      self.gradBias = torch.CudaTensor()
   end

   -- Initialize from input on the first updateOutput / updateGradInput
   self.output = nil
   self.gradInput = nil
end

function BN:reset()
   self.weight:uniform()
   self.bias:zero()
end

function BN:updateOutput(input)
   assert(input:dim() == 2, 'only mini-batch supported (2D tensor), got '
             .. input:dim() .. 'D tensor instead')

   self.std = self.std or self.running_std:clone():zero():cuda()
   self.std:resizeAs(self.running_std)
   self.centered = self.centered or input:clone():zero():cuda()
   self.centered:resizeAs(input)
   self.normalized = self.normalized or input:clone():zero():cuda()
   self.normalized:resizeAs(input)
   self.output = self.output or input:clone():zero():cuda()
   self.output:resizeAs(input)

   BNFFI.BatchNormalizationUpdateOutputFFI(cutorch._state,
                                           input:cdata(),
                                           self.output:cdata(),
                                           self.centered:cdata(),
                                           self.std:cdata(),
                                           self.normalized:cdata(),
                                           self.running_mean:cdata(),
                                           self.running_std:cdata(),
                                           self.weight:cdata(),
                                           self.bias:cdata(),
                                           self.eps,
                                           self.momentum,
                                           self.train,
                                           self.affine)

   return self.output
end

function BN:updateGradInput(input, gradOutput)
   assert(input:dim() == 2, 'only mini-batch supported')
   assert(gradOutput:dim() == 2, 'only mini-batch supported')
   assert(self.train == true,
          'should be in training mode when self.train is true')

   self.gradInput = self.gradInput or input:clone():zero():cuda()
   self.gradInput:resizeAs(input)

   BNFFI.BatchNormalizationUpdateGradInputFFI(cutorch._state,
                                              self.gradInput:cdata(),
                                              gradOutput:cdata(),
                                              self.centered:cdata(),
                                              self.std:cdata(),
                                              self.weight:cdata(),
                                              self.affine)

   return self.gradInput
end

function BN:accGradParameters(input, gradOutput, scale)
   if self.affine then
      scale = scale or 1.0
      BNFFI.BatchNormalizationAccGradParametersFFI(cutorch._state,
                                                   gradOutput:cdata(),
                                                   self.normalized:cdata(),
                                                   self.gradWeight:cdata(),
                                                   self.gradBias:cdata(),
                                                   scale)
   end

end

function BN:clearState()
   self.centered = nil
   self.std = nil
   self.normalized = nil

   parent.clearState(self)
end

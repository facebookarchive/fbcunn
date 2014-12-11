-- Copyright 2004-present Facebook. All Rights Reserved.
--
-- LocallyConnected layer, see
-- https://code.google.com/p/cuda-convnet/wiki/LayerParams#Locally-connected_layer_with_unshared_weights
--
local LocallyConnected, parent = torch.class('nn.LocallyConnected',
                                             'nn.Module')

function LocallyConnected:__init(nInputPlane, iW, iH, nOutputPlane, kW, kH,
                                 dW, dH)
   parent.__init(self)

   dW = dW or 1
   dH = dH or 1

   -- validate inputs
   assert(nInputPlane > 0, "Number of input planes must be positive.")
   assert(iW > 0, "Input image width must be positive.")
   assert(iH > 0, "Input image height must be positive.")
   assert(nOutputPlane > 0, "Number of output planes must be positive.")
   assert(0 < kW, "Kernel width must be positive.")
   assert(0 < kH, "Kernel height must be positive.")
   assert(0 < dW, "Column stride must be positive.")
   assert(0 < dH, "Row stride must be positive.")
   assert(kW <= iW, "Kernel width must not exceed input image width.")
   assert(kH <= iH, "Kernel height must not exceed input image height.")

   -- initialize module state
   self.nInputPlane = nInputPlane
   self.iW = iW
   self.iH = iH
   self.nOutputPlane = nOutputPlane
   self.kW = kW
   self.kH = kH

   self.dW = dW
   self.dH = dH

   local oW, oH = self:outputSize()
   self.weight = torch.Tensor(nOutputPlane, oH, oW, nInputPlane, kH, kW)
   self.bias  = torch.Tensor(nOutputPlane)
   self.gradWeight = torch.Tensor():resizeAs(self.weight)
   self.gradBias = torch.Tensor():resizeAs(self.bias)

   self.output = torch.Tensor()

   self:reset()
end

function LocallyConnected:outputSize()
   local oW = math.floor((self.iW - self.kW) / self.dW + 1)
   local oH = math.floor((self.iH - self.kH) / self.dH + 1)

   return oW, oH
end

function LocallyConnected:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1/math.sqrt(self.kW*self.kH*self.nInputPlane)
   end
   self.weight:uniform(-stdv, stdv)
   self.bias:uniform(-stdv, stdv)
end

function LocallyConnected:updateOutput(input)
   -- validate inputs
   assert(input:dim() == 3 or input:dim() == 4,
          "Invalid input. Must be 3- or 4-D")

   if input:dim() == 3 then
      assert(input:size(1) == self.nInputPlane,
             "Number of input planes mismatch")
      assert(input:size(2) == self.iH, "Input height mismatch")
      assert(input:size(3) == self.iW, "Input width mistmatch")
   else
      assert(input:size(2) == self.nInputPlane,
             "Number of input planes mismatch")
      assert(input:size(3) == self.iH, "Input height mismatch")
      assert(input:size(4) == self.iW, "Input width mismatch")
   end

   -- resize output based on configuration
   -- this can't be done in the constructor because we don't bake
   -- batch size into the layer state. (perf note: tensor resize to same size
   -- is a no-op.)
   local size = input:size()
   local oW, oH = self:outputSize()
   if (input:dim() == 3) then
      size[1] = self.nOutputPlane
      size[2] = oH
      size[3] = oW
   else
      size[2] = self.nOutputPlane
      size[3] = oH
      size[4] = oW
   end
   self.output = self.output:resize(size)

   return input.nn.LocallyConnected_updateOutput(self, input)
end

function LocallyConnected:updateGradInput(input, gradOutput)
   return input.nn.LocallyConnected_updateGradInput(self, input, gradOutput)
end

function LocallyConnected:accGradParameters(input, gradOutput, scale)
   scale = scale or 1.0
   return input.nn.LocallyConnected_accGradParameters(self, input, gradOutput,
                                                      scale)
end

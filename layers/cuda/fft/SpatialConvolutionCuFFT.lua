-- Copyright 2004-present Facebook. All Rights Reserved.

local SpatialConvolutionCuFFT, parent =
      torch.class('nn.SpatialConvolutionCuFFT', 'nn.Module')

local precision = 0.00002
local printDebug = false

function SpatialConvolutionCuFFT:__init(nInputPlane, nOutputPlane,
                                        kW, kH, dW, dH, debug, printDebug)
   parent.__init(self)

   self.nInputPlane = nInputPlane
   self.nOutputPlane = nOutputPlane
   self.kW = kW
   self.kH = kH
   self.dW = dW or 1
   self.dH = dH or 1
   self.debug = debug or false
   self.printDebug = printDebug or false

   assert(self.dW == 1, "fft only supports stride-1 convolutions atm")

   self.weight = torch.Tensor(nOutputPlane, nInputPlane, kH, kW)
   self.bias = torch.Tensor(nOutputPlane)
   self.gradWeight = torch.Tensor(nOutputPlane, nInputPlane, kH, kW)
   self.gradBias = torch.Tensor(nOutputPlane)

   self:reset()
end

function SpatialConvolutionCuFFT:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1/math.sqrt(self.kW*self.kH*self.nInputPlane)
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

local function debugVSMM(pass, module, toTest, fun, param1, param2, param3)
   local o = toTest:float():clone()
   toTest:zero()
   module.padding = 0
   module.finput = torch.CudaTensor()
   module.fgradInput = torch.CudaTensor()
   -- linearize weight for MM
   module.gradWeight =
      module.gradWeight:view(module.nOutputPlane,
                             module.nInputPlane * module.kH * module.kW)
   module.weight =
      module.weight:view(module.nOutputPlane,
                         module.nInputPlane * module.kH * module.kW)
   local test = fun(module, param1, param2, param3)
   -- reset layout of weight after MM
   module.gradWeight =
      module.gradWeight:view(module.nOutputPlane,
                             module.nInputPlane,
                             module.kH,
                             module.kW)
   module.weight =
      module.weight:view(module.nOutputPlane,
                         module.nInputPlane,
                         module.kH,
                         module.kW)
   local norm = math.sqrt(test:float():dot(test:float()) + 1e-8)
   if test:float():dist(o:float()) / norm > precision then
      print('error ', pass, test:float():dist(o:float()) / norm, precision)
      os.exit()
   elseif printDebug then
      print('debug vs MM check passes ',
            pass, o:min(), o:max(), o:mean(), o:std(), o:sum())
   end
end

function SpatialConvolutionCuFFT:updateOutput(input)
   input.nn.SpatialConvolutionCuFFT_updateOutput(self, input)

   if self.debug then
      debugVSMM("updateOutput",
                self,
                self.output,
                input.nn.SpatialConvolutionMM_updateOutput,
                input)
   end

   return self.output
end

function SpatialConvolutionCuFFT:explorePerformance(input, batches,
      inputs, planes, inputRows, inputCols, kernelRows, kernelCols)
   input.nn.SpatialConvolutionCuFFT_explorePerformance(self, batches,
      inputs, planes, inputRows, inputCols, kernelRows, kernelCols)
end

function SpatialConvolutionCuFFT:cleanupBuffers(input)
   input.nn.SpatialConvolutionCuFFT_cleanupBuffers()
end

function SpatialConvolutionCuFFT:updateGradInput(input, gradOutput)
   input.nn.SpatialConvolutionCuFFT_updateGradInput(self, gradOutput)

   if self.debug then
      debugVSMM("updateGradInput",
                self,
                self.gradInput,
                input.nn.SpatialConvolutionMM_updateGradInput,
                input,
                gradOutput)
   end

   return self.gradInput
end

local
function wrapMM_accGradParameters_gradWeight(module, input, gradOutput, scale)
   input.nn.SpatialConvolutionMM_accGradParameters(
      module, input, gradOutput, scale)
   return module.gradWeight
end

local
function wrapMM_accGradParameters_gradBias(module, input, gradOutput, scale)
   input.nn.SpatialConvolutionMM_accGradParameters(
      module, input, gradOutput, scale)
   return module.gradBias
end

function SpatialConvolutionCuFFT:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   input.nn.SpatialConvolutionCuFFT_accGradParameters(
     self, input, gradOutput, scale)

   if self.debug then
      self.gradBias:zero() -- zero first to avoid accumulation
      debugVSMM("accGradParameters_gradWeight",
                self,
                self.gradWeight,
                wrapMM_accGradParameters_gradWeight,
                input,
                gradOutput,
                scale)
      local saveBias = self.gradBias:float():clone()
      self.gradWeight:zero()
      self.gradBias:zero()
      debugVSMM("accGradParameters_gradBias",
                self,
                saveBias,
                wrapMM_accGradParameters_gradBias,
                input,
                gradOutput,
                scale)
   end
end

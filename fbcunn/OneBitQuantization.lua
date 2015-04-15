-- Copyright 2004-present Facebook. All Rights Reserved.

require 'cutorch'
require 'nn'

--[[
CUDA implementation of the quantize/unquantize methods used by `nn.OneBitDataParallel`.
]]
local OneBitQuantization = torch.class('nn.OneBitQuantization')

function OneBitQuantization:__init()
   self.quantized = torch.CudaTensor()
   self.non_quantized = torch.CudaTensor()
   self.quantization_error = nil
   self.avg_pos = torch.CudaTensor()
   self.avg_neg = torch.CudaTensor()
end

function OneBitQuantization:reset()
   self.quantization_error = nil
end

function OneBitQuantization:quantize(non_quantized_input)
   -- When starting a new quantization chain, we start with zero error
   if not self.quantization_error then
      self.quantization_error = non_quantized_input:clone()
      self.quantization_error:zero()
   end

   non_quantized_input.nn.OneBitQuantization_quantize(self, non_quantized_input)
   return self.quantized, self.quantization_error, self.avg_pos, self.avg_neg
end

function OneBitQuantization:dequantize(quantized_input,
                                       avg_pos, avg_neg, num_orig_cols)
   quantized_input.nn.OneBitQuantization_dequantize(
      self, quantized_input, avg_pos, avg_neg, num_orig_cols)
   return self.non_quantized
end

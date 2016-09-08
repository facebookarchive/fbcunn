-- Copyright 2014 - present Facebook. All Rights Reserved.

-- This is the module that you should most likely call if you want the fastest
-- convolution available. It is a wrapper to cudnn as well as different
-- FFT-based implementations.
--
-- Instantiate with fbnn.SpatialConvolution(nInputPlane,
--                                          nOutputPlane,
--                                          kW,
--                                          kH,
--                                          dW,                    [1]
--                                          dH,                    [1]
--                                          padLeft,               [0]
--                                          padUp,                 [0]
--                                          maximalMemoryOverhead, [nil]
--                                          inferenceOnly)         [false]
-- where:
--   - the first parameters have the traditional meaning,
--   - maximalMemoryOverhead: limit on the amount of memory
--     overhead you want to allow, nil meaning no limit
--   - inferenceOnly: whether the module is used for inference or training.
--     Spercifying inference only saves time in the autotuning process
--
-- On the first call to updateOutput, a simple autotuning search kicks off
-- which compares the performance of different flavors of:
--   FBFFT + FBMM, FBFFT + cublasGemm, FBFFT Tiled sync, FBFFT Tiled async
--   and cudnn
-- In the future we can also wrap more specialized kernels (e.g.
--   no memory overhead FFTs, Nervana's convolutions etc)

require 'cudnn'

local argcheck = require 'argcheck'
local SpatialConvolution, parent =
   torch.class('fbnn.SpatialConvolution', 'nn.Module')

fbnn.SpatialConvolution.reportErrors = false
fbnn.SpatialConvolution.reportWarnings = false

function SpatialConvolution:__init(nInputPlane,
                                   nOutputPlane,
                                   kW,
                                   kH,
                                   dW,
                                   dH,
                                   padLeft,
                                   padUp,
                                   maximalMemoryOverhead,
                                   inferenceOnly)
   parent.__init(self)
   self.inputPlanes = nInputPlane
   self.outputPlanes = nOutputPlane
   self.kW = kW
   self.kH = kH
   self.dW = dW or 1
   self.dH = dH or 1
   self.padLeft = padLeft or 0
   self.padUp = padUp or 0
   self.inferenceOnly = inferenceOnly
   self.maximalMemoryOverhead = maximalMemoryOverhead
   self.reportLevel = 0

   -- Allocate an underlying CuDNN
   self.cudnnModuleInst =
      cudnn.SpatialConvolution(nInputPlane,
                               nOutputPlane,
                               kW,
                               kH,
                               dW,
                               dH,
                               padLeft,
                               padUp):cuda()

   -- Take its tensors as my own
   self.weight = self.cudnnModuleInst.weight
   self.output = self.cudnnModuleInst.output
   self.bias = self.cudnnModuleInst.bias
   self.gradWeight = self.cudnnModuleInst.gradWeight
   self.gradBias = self.cudnnModuleInst.gradBias
end

function SpatialConvolution:setInferenceOnly(val)
   assert(type(val) == 'boolean')
   self.inferenceOnly = val
end

function SpatialConvolution:setReuseWeights(val)
   assert(self.bestModuleInst, 'Must tune before reusing weights')
   if self.bestModuleInst.setReuseWeights then
      self.bestModuleInst:setReuseWeights(val)
   end
end

--------------------------------------------------------------------------------
-- Detail
--------------------------------------------------------------------------------
local function _timeFunction(
      fun, mod, arg1, arg2, arg3, arg4, arg5)
   local numTrials = 3
   local time = 0
   cutorch.synchronize()
   for i = 1, numTrials do
      local timer = torch.Timer()
      fun(mod, arg1, arg2, arg3, arg4, arg5)
      cutorch.synchronize()
      if i > 1 then
         time = time + timer:time().real
      end
   end
   time = time / (numTrials - 1)
   return time * 1000
end

local runModule = argcheck {
   { name = "mod", type = "table" },
   -- { name = "mod", type = "nn.Module" },
   -- { name = "mod", type = "nn.SpatialConvolutionFBFFT" },
   { name = "input", type = "torch.CudaTensor"},
   { name = "gradOutput", type = "torch.CudaTensor"},
   { name = "parameters", type = "table"},
   { name = "extraParameters", type = "table"},
   { name = "inferenceOnly", type = "boolean"},
   { name = "scale", type = "number"},
   call = function(
      mod, input, gradOutput, parameters, extraParameters, inferenceOnly, scale)
         local params = {}
         for _, v in pairs(parameters) do
            table.insert(params, v)
         end
         for _, v in pairs(extraParameters) do
            table.insert(params, v)
         end

         local inst = mod(unpack(params)):cuda()

         -- Setup autotuning behavior, unused in CuDNN
         inst.printDebugLevel = -1
         if inst.printDebugLevel >= 3 then
            print(inst, unpack(params))
            inst.cudnnDebug = true
            if inst.printDebugLevel >= 4 then
               input:fill(1.0)
               inst.weight:fill(1.0)
               gradOutput:fill(1.0)
            else
               input:normal()
               inst.weight:normal()
               gradOutput:normal()
            end
         end
         inst.autotuningPass = true
         inst.reportErrors = fbnn.SpatialConvolution.reportErrors or false

         local timing1, timing2, timing3 = 0, 0, 0
         timing1 = timing1 +
            _timeFunction(inst.updateOutput, inst, input)
         if not inst.success then
            inst:cleanupBuffers()
            return 1e32, 0, 0, nil
         end

         if inferenceOnly then
            return timing1, 0, 0, inst
         end

         timing2 = timing2 +
            _timeFunction(inst.updateGradInput, inst, input, gradOutput)
         if not inst.success then
            inst:cleanupBuffers()
            return 1e32, 0, 0, nil
         end

         timing3 = timing3 +
            _timeFunction(inst.accGradParameters, inst, input, gradOutput, scale)
         if not inst.success then
            inst:cleanupBuffers()
            return 1e32, 0, 0, nil
         end

         -- Unset autotuning behavior, unused in CuDNN
         inst.autotuningPass = false
         inst.reportErrors = true

         return timing1, timing2, timing3, inst
      end
}

function SpatialConvolution:_tune(batchSize,
                                  iW,
                                  iH,
                                  nInputPlane,
                                  nOutputPlane,
                                  kW,
                                  kH,
                                  dW,
                                  dH,
                                  padLeft,
                                  padUp,
                                  inferenceOnly)
   -- Just compare cudnn to various FFT variants and pick the best
   local timings = {}
   local ps = {batchSize, nInputPlane, iH, iW}
   local input = torch.Tensor(torch.LongStorage(ps)):cuda()
   local ps = {batchSize,
               nOutputPlane,
               math.floor((iH - kH + 2 * padUp) / dH) + 1,
               math.floor((iW - kW + 2 * padLeft) / dW) + 1}
   local gradOutput = torch.Tensor(torch.LongStorage(ps)):cuda()
   local scale = torch.random(100) / 100.0

   local preFree = cutorch.getMemoryUsage()
   local timing1, timing2, timing3 = 0, 0, 0
   timing1 = timing1 + _timeFunction(self.cudnnModuleInst.updateOutput,
                          self.cudnnModuleInst,
                          input)
   if not inferenceOnly then
      timing2 = timing2 + _timeFunction(self.cudnnModuleInst.updateGradInput,
                             self.cudnnModuleInst,
                             input,
                             gradOutput)
      timing3 = timing3 + _timeFunction(self.cudnnModuleInst.accGradParameters,
                             self.cudnnModuleInst,
                             input,
                             gradOutput,
                             scale)
   end
   local postFree = cutorch.getMemoryUsage()
   local cudnnTiming = timing1 + timing2 + timing3
   timings[self.cudnnModuleInst] = {
      parameters = nil,
      memoryConsumption = preFree - postFree,
      timing1,
      timing2,
      timing3
   }

   -- Only investigate FFT for stride == 1
   local bestTiming = 1e32
   if dW == 1 and dH == 1 then
      local bestModule = nil
      self.bestModuleInst = nil
      local modules

      if iW > 32 or iH > 32 then
         -- Don't waste time on inefficient 64x64 or 128x128 convolutions atm
         -- TODO: Fix 3 issues:
         --   1. implement fast 64 and 128,
         --   2. drop buffer malloced at each call
         --   3. tune FBMM for 64x64 and 128x128
         modules = {
            -- requires explicit padding and is slow
            -- nn.SpatialConvolutionCuFFT,
            nn.SpatialConvolutionFFTTiledSync,
            nn.SpatialConvolutionFFTTiledAsync,
            -- too slow atm
            -- nn.SpatialConvolutionFFTTiledIterated
         }
      else
         modules = {
            -- requires explicit padding and is slow
            -- nn.SpatialConvolutionCuFFT,
            nn.SpatialConvolutionFBFFT,
            -- only activate if fbmm perf is suspiciously low
            -- nn.SpatialConvolutionFBFFTGemm, activate if suspicious fbmm perf
            nn.SpatialConvolutionFFTTiledSync,
            nn.SpatialConvolutionFFTTiledAsync,
            -- too slow atm
            -- nn.SpatialConvolutionFFTTiledIterated
         }
      end

      for i_mod in pairs(modules)
      do
         local mod = modules[i_mod]
         local extraParameters = {}
         if mod == nn.SpatialConvolutionFBFFT or
            mod == nn.SpatialConvolutionFBFFTGemm
         then
            extraParameters = {
               -- reuse, streams
               {nn.SpatialConvolutionFFT.memoryReuseAll, 16},
               {nn.SpatialConvolutionFFT.memoryReuseNone, 16}
            }
         elseif mod == nn.SpatialConvolutionFFTTiledSync
            or mod == nn.SpatialConvolutionFFTTiledAsync
            or mod == nn.SpatialConvolutionFFTTiledIterated
         then
            -- tileH, tileW, reuse
            if kH <= 3 and kW <= 3 then
               extraParameters = {
                  -- Only enable 8 x 8 manually, is often too expensive by default
                  -- {8, 8, nn.SpatialConvolutionFFT.memoryReuseNone},
                  {16, 16, nn.SpatialConvolutionFFT.memoryReuseNone},
                  {32, 32, nn.SpatialConvolutionFFT.memoryReuseNone},
                  -- {8, 8, nn.SpatialConvolutionFFT.memoryReuseAll},
                  {16, 16, nn.SpatialConvolutionFFT.memoryReuseAll},
                  {32, 32, nn.SpatialConvolutionFFT.memoryReuseAll},
               }
            elseif kH <= 9 and kW <= 9 then
               extraParameters = {
                  {16, 16, nn.SpatialConvolutionFFT.memoryReuseNone},
                  {32, 32, nn.SpatialConvolutionFFT.memoryReuseNone},
                  {16, 16, nn.SpatialConvolutionFFT.memoryReuseAll},
                  {32, 32, nn.SpatialConvolutionFFT.memoryReuseAll},
               }
            else
               extraParameters = {
                  {32, 32, nn.SpatialConvolutionFFT.memoryReuseNone},
                  {32, 32, nn.SpatialConvolutionFFT.memoryReuseAll},
               }
            end
         end

         for i_params in pairs(extraParameters)
         do
            local preFree = cutorch.getMemoryUsage()
            local timing1, timing2, timing3, inst =
               runModule(mod,
                         input,
                         gradOutput,
                         { nInputPlane,
                           nOutputPlane,
                           kW,
                           kH,
                           dW,
                           dH,
                           padLeft,
                           padUp
                         },
                         extraParameters[i_params],
                         inferenceOnly,
                         scale
               )

            local postFree = cutorch.getMemoryUsage()
            local exceedsAdmissibleMemory = true
            if inst then
               timings[inst] = {
                  parameters = extraParameters[i_params],
                  memoryConsumption = preFree - postFree,
                  timing1,
                  timing2,
                  timing3
               }
               exceedsAdmissibleMemory =
                  (self.maximalMemoryOverhead and
                      (timings[inst].memoryConsumption -
                          timings[self.cudnnModuleInst].memoryConsumption) >
                      self.maximalMemoryOverhead)

            end

            if timing1 + timing2 + timing3 < bestTiming and
               not exceedsAdmissibleMemory
            then
               bestTiming = timing1 + timing2 + timing3
               bestModule = mod
               if self.bestModuleInst and self.bestModuleInst.cleanupBuffers then
                  self.bestModuleInst:cleanupBuffers()
               end
               self.bestModuleInst = inst
            elseif inst then
               inst:cleanupBuffers()
            end
            inst = nil
            collectgarbage()
            collectgarbage()
         end
      end

      if self.reportLevel >= 3 then
         print('Timings: ', timings)
      end
      if self.reportLevel >= 1 then
         print('Best FFT: ', bestTiming, ' ', self.bestModuleInst)
         print('cudnn   : ', cudnnTiming, ' ', self.cudnnModuleInst)
      end
      if self.reportLevel >= 2 then
         print('FFT   detail ', timings[self.bestModuleInst])
         print('CuDNN detail ', timings[self.cudnnModuleInst])
      end

      -- Always run correctness check atm, move later to only run when FFT wins.
      if bestModule ~= cudnn.SpatialConvolution and self.bestModuleInst then
         -- Fail if check fails here, don't fallback to cudnn
         self.bestModuleInst.autotuningPass = true
         self.bestModuleInst.cudnnDebug = true
         self.bestModuleInst.printDebugLevel = -1
         input:normal()
         gradOutput:normal()
         self.bestModuleInst:reset()
         self.bestModuleInst:updateOutput(input)
         if not inferenceOnly then
            self.bestModuleInst:updateGradInput(input, gradOutput)
            self.bestModuleInst:accGradParameters(input, gradOutput, scale)
         end
         assert(self.bestModuleInst.cudnnChecks)
         self.bestModuleInst.autotuningPass = false
         self.bestModuleInst.cudnnDebug = false
         self.bestModuleInst.printDebugLevel = -1
      end
   end

   if bestTiming > cudnnTiming then
      self.bestModuleInst = self.cudnnModuleInst
      self.bestModuleInst:resetWeightDescriptors()
   end

   -- if self.bestModuleInst == self.cudnnModuleInst, just reduces the refcount
   -- otherwise prepares for collection
   self.cudnnModuleInst = nil

   -- Take as my own
   self.weight = self.bestModuleInst.weight
   self.output = self.bestModuleInst.output
   self.bias = self.bestModuleInst.bias
   self.gradWeight = self.bestModuleInst.gradWeight
   self.gradBias = self.bestModuleInst.gradBias

   collectgarbage()
   collectgarbage()
end

-- Update output (i.e. forward prop)
function SpatialConvolution:updateOutput(input)
   assert(#input:size() == 4, 'Only supports 4-D tensors atm')

   if not self.bestModuleInst then
      -- used for tuning consistency
      self.batchSize = input:size(1)
      self.iH = input:size(3)
      self.iW = input:size(4)
      self:_tune(self.batchSize,
                 self.iW,
                 self.iH,
                 self.inputPlanes,
                 self.outputPlanes,
                 self.kW,
                 self.kH,
                 self.dW,
                 self.dH,
                 self.padLeft,
                 self.padUp,
                 self.inferenceOnly)
   end

   assert(self.batchSize == input:size(1),
          'Batches tuned for: ' .. self.batchSize .. ' VS ' ..  input:size(1))
   assert(self.inputPlanes == input:size(2),
          'InputPlanes tuned for: ' .. self.inputPlanes ..
             ' VS ' ..  input:size(2))
   assert(self.iH == input:size(3),
          'InputH tuned for: ' .. self.iH .. ' VS ' ..  input:size(3))
   assert(self.iW == input:size(4),
          'InputW tuned for: ' .. self.iW .. ' VS ' ..  input:size(4))

   -- weights are updated each iteration, pass them on
   self.bestModuleInst.weight = self.weight
   self.output = self.bestModuleInst:updateOutput(input)
   self.bias = self.bestModuleInst.bias

   assert(self.outputPlanes == self.output:size(2),
          'OutputPlanes tuned for: ' .. self.outputPlanes ..
             ' VS ' ..  self.output:size(2))

   assert(self.bestModuleInst)
   if torch.type(self.bestModuleInst) ~= 'cudnn.SpatialConvolution' then
      assert(self.bestModuleInst.cudnnChecks)
   end

   return self.output
end


function SpatialConvolution:updateGradInput(input, gradOutput)
   assert(self.bestModuleInst, 'Must have been tuned in updateOutput already!')
   assert(not self.inferenceOnly, 'Inference only specified => no gradInput ')
   self.bestModuleInst.gradInput =
      self.bestModuleInst:updateGradInput(input, gradOutput)
   self.gradInput = self.bestModuleInst.gradInput
   return self.gradInput
end


function SpatialConvolution:accGradParameters(
      input, gradOutput, scale)
   assert(self.bestModuleInst, 'Must have been tuned in updateOutput already!')
   assert(not self.inferenceOnly, 'Inference only specified => no accGrads ')
   -- gradWeight / gradBias are updated each iteration, pass them on
   self.bestModuleInst.gradWeight = self.gradWeight
   self.bestModuleInst.gradBias = self.gradBias
   self.bestModuleInst:accGradParameters(input, gradOutput, scale)
end


function SpatialConvolution:cleanupBuffers()
   if self.bestModuleInst and self.bestModuleInst.cleanupBuffers then
      self.bestModuleInst:cleanupBuffers()
   end
   self.bestModuleInst = nil
end

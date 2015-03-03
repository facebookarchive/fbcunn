-- Copyright 2004-present Facebook. All Rights Reserved.
-- [[This file defines an example class]]
require('cutorch')
local withDevice = cutorch.withDevice
local dprintL = (require 'fb.util.dbg').new('parallel')
local dprint = function(...)
    return dprintL(1, ...)
end
local pl = require('pl.import_into')()

--[[
DataParallel splits the input along separate columns, that run the
same models on distinct partitions of the input.

Pictorially
```
                        +--------+
        column 1        |        |         column 3
           +------------+  Input +-------------+
           |            |        |             |
           |            +----+---+             |
           |                 |                 |
           |                 |                 |
      +----+---+        +----+---+        +----+---+
      |        |        |        |        |        |
      | Linear |        | Linear |        | Linear |       row 1
      |        |        |        |        |        |
      +----+---+        +----+---+        +----+---+
           |                 |                 |
           |                 |                 |
      +----+---+        +----+---+        +----+---+
      |        |        |        |        |        |
      |  Tanh  |        |  Tanh  |        |  Tanh  |       row 2
      |        |        |        |        |        |
      +----+---+        +----+---+        +----+---+
           |                 |                 |
           |                 |                 |
           |                 |                 |
           |            +----+---+             |
           |            |        |             |
           +------------+ Output +-------------+
                        |        |
                        +--------+
```
]]
local DataParallel, _ = torch.class('nn.DataParallel',
                                    'nn.AbstractParallel')

-- `_distributeInput` slices the input along self.dimension
-- and copies each portion into each child module.
function DataParallel:_distributeInput(input)
    local container_gpuid = cutorch.getDevice()
    local outerDim = input:size(self.dimension)
    if outerDim % #self.modules ~= 0 then
        error("cannot evenly divide " .. outerDim .. " inputs to " ..
              #self.modules .. " modules")
    end
    local eltsPerMod = outerDim / #self.modules

    local function inputSlice(i)
        local rangeStart = (i - 1) * eltsPerMod + 1
        local retval = input:narrow(self.dimension, rangeStart, eltsPerMod)
        return retval
    end

    assert(torch.typename(input) == 'torch.CudaTensor')
    for i, module in ipairs(self.modules) do
        local gpuid = self.gpu_assignments[i]
        withDevice(gpuid, function()
            local slice = inputSlice(i)
            if gpuid == container_gpuid then
                self.input_gpu[gpuid] = slice
                return
            end
            self.input_gpu[gpuid] = self.input_gpu[gpuid] or torch.CudaTensor()
            self.input_gpu[gpuid]:resizeAs(slice)
            self:gpuSend(self.input_gpu[gpuid], slice)
        end)
    end
end

-- `_mixGrads` applies the _combineGradients operator
-- for each row in the DataParallel module.
function DataParallel:_mixGrads()
   local gradients = {}
   for i=1,#self.modules do
      local _,g = self.modules[i]:parameters()
      gradients[i] = g
   end
   -- if no parameters then do nothing
   if #gradients == 0 or #gradients[1] == 0 then return end

   -- for each entry in "parameters",
   -- create a table with the equivalent parameters
   -- from all GPUs for that entry and send it to _combine_gradients
   for i=1,#gradients[1] do
      local t = {}
      for j=1,#gradients do
         t[j] = gradients[j][i]
      end
      self:_combine_gradients(i, t)
   end
end

function DataParallel:_combine_gradients(row, gradients)
   local homeTensor = gradients[1]
   local homeDevice = gradients[1]:getDevice()

    self.homeGradBuffers = self.homeGradBuffers or {}
    self.homeGradBuffers[row] = self.homeGradBuffers[row] or {}
    local homeGradBuffer = self.homeGradBuffers[row]

    -- First compute the sum; copy everything onto one GPU
    -- and add it up.
    withDevice(homeDevice, function()
       -- put in separate for-loops so that the GPU gathers are overlapped
       for j = 2, #gradients do
          homeGradBuffer[j] = homeGradBuffer[j] or homeTensor.new()
          homeGradBuffer[j]:resizeAs(homeTensor)
          self:gpuSend(homeGradBuffer[j], gradients[j])
       end
       for j = 2, #gradients do
          homeTensor:add(homeGradBuffer[j])
       end
    end)

    -- Now copy it back to each GPU
    for j = 2, #gradients do
        self:gpuSend(gradients[j], gradients[1])
    end
end

function DataParallel:name()
    return 'DataParallel'
end

function DataParallel:__tostring__()
   return self:name() .. '\n' .. self.modules[1]:__tostring__()
end

function DataParallel:updateGradInput(_input, gradOutput)
   self:_distributeGradOutput(_input, gradOutput)

    -- update gradInput for each module
    for i,module in ipairs(self.modules) do
        local gpuid = self.gpu_assignments[i]
        withDevice(gpuid, function()
            module:updateGradInput(self.input_gpu[gpuid],
                                   self.gradOutput_gpu[i])
        end)
    end

    -- gradInput for each module on its appropriate gpu
    if not self.gradInput then return end -- if gradInput is nil, do nothing
    self.gradInput:resizeAs(self.input_gpu[self.container_gpuid])

    self.gradInput:resizeAs(_input)
    local elementsPerSlice = self.input_gpu[1]:size(self.dimension)
    -- add gradInputs
    for i, module in ipairs(self.modules) do
        if module.gradInput then
           local parent_gpuid = self.gpu_assignments[i]
           withDevice(parent_gpuid, function()
                         self.gradInput_gpu[i] = self.gradInput_gpu[i]
                            or torch.CudaTensor()
                         self.gradInput_gpu[i]:resizeAs(module.gradInput)
                         self:gpuSend(self.gradInput_gpu[i], module.gradInput)
                         self.gradInput:narrow(
                            self.dimension,
                            (i - 1) * elementsPerSlice + 1,
                            elementsPerSlice):copy(self.gradInput_gpu[i]
                                                  )
           end)
        end
    end

    return self.gradInput
end

function DataParallel:accUpdateGradParameters(_input, _gradOutput, lr)
   -- to implement this, you have to write a function called _mixWeights, that
   -- like mixGrads, averages the weights across all GPUs
   error('accUpdateGradParameters not implemented for: ' .. torch.type(self))
end

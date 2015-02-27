-- Copyright 2004-present Facebook. All Rights Reserved.
-- [[This file defines an example class]]
require('cutorch')
local withDevice = cutorch.withDevice
local dprint = function(...)
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
    dprint("_distributeInput: ", input)
    local outerDim = input:size()[self.dimension]
    if outerDim % #self.modules ~= 0 then
        error("cannot evenly divide " .. outerDim .. " inputs to " ..
              #self.modules .. " modules")
    end
    local eltsPerMod = outerDim / #self.modules

    local function inputSlice(i)
        local rangeStart = (i - 1) * eltsPerMod + 1
        local rangeEnd = rangeStart + eltsPerMod - 1
        local retval = input[{ {rangeStart, rangeEnd} }]
        dprintL(5, "inputSlice", i, {rangeStart, rangeEnd}, retval )
        return retval
    end

    assert(torch.typename(input) == 'torch.CudaTensor')
    for i, module in ipairs(self.modules) do
        local gpuid = self.gpu_assignments[i]
        dprintL(2, "collecting module for gpu", gpuid)
        withDevice(gpuid, function()
            local slice = inputSlice(i)
            assert(torch.typename(slice) == 'torch.CudaTensor')
            if gpuid == container_gpuid then
                dprintL(2, "already on container_gpuid!", self.input_gpu, gpuid)
                self.input_gpu[gpuid] = slice
                return
            end
            dprintL(2, "setting remote gpuid!", self.input_gpu[gpuid], gpuid)
            self.input_gpu[gpuid] = self.input_gpu[gpuid] or torch.CudaTensor()
            self.input_gpu[gpuid]:resizeAs(slice)
            self:gpuSend(self.input_gpu[gpuid], slice)
        end)
    end
    dprint("after DataParallel:_distributeInput", self.input_gpu)
end

-- `_mixGrads` applies the `_combineAcrossColumns` operator (e.g. averaging)
-- for each row in the DataParallel module.
function DataParallel:_mixGrads()
    -- [column][submodule][grads]
    local subModToGrads = pl.tablex.map(
        function(module)
            local submod_grads = {}
            module:for_each(
                function(submod)
                    local params, grads = submod:parameters()
                    if params then
                        assert(grads)
                        assert(#grads > 1)
                        table.insert(submod_grads, grads)
                    end
                end
            )
            return submod_grads
        end,
        self.modules
    )

    -- [submodule][column][grads]
    local gradsPerSubMod = pl.tablex.zip(table.unpack(subModToGrads))

    -- Now combine them all
    pl.tablex.foreachi(
        gradsPerSubMod,
        function(row, row_idx)
            return self:_combineAcrossColumns(row_idx, row)
        end
    )
end

function DataParallel:_combine_gradients(row_idx, grad_idx, gradients)
    local homeTensor = gradients[1]
    local homeDevice = gradients[1]:getDevice()

    -- Build a table of the tensors on each GPU.
    local tempTensors = { homeTensor }
    -- First compute the average; copy everything onto one GPU
    -- and add it up.
    withDevice(homeDevice, function()
                   for j = 2, #gradients do
                       table.insert(tempTensors,
                                    torch.CudaTensor(homeTensor:size()))
                       self:gpuSend(tempTensors[#tempTensors], gradients[j])
                   end
    end)
    for j = 2, #tempTensors do
        dprint("about to add", j, gradients[1], tempTensors[j])
        gradients[1]:add(tempTensors[j])
    end
    gradients[1]:div(#gradients)

    -- Now copy it everywhere
    for j = 2, #gradients do
        self:gpuSend(gradients[j], gradients[1])
    end
end

-- This helper destructively averages N parallel tables of tensors,
-- possibly on different GPUs.  All the tensors are modified in place.
function DataParallel:_combineAcrossColumns(row_idx, row_gradients)
    assert(row_gradients)
    assert(#row_gradients >= 1)
    assert(#row_gradients[1] >= 1)
    assert(type(row_gradients[1]) == 'table')
    assert(type(row_gradients) == 'table')
    assert(torch.typename(row_gradients[1][1]) == 'torch.CudaTensor')

    for grad_idx = 1,#row_gradients[1] do
        local gradients =
            pl.tablex.map(function(sg) return sg[grad_idx] end, row_gradients)
        self:_combine_gradients(row_idx, grad_idx, gradients)
    end
end

function DataParallel:name()
    return 'DataParallel'
end

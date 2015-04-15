-- Copyright 2004-present Facebook. All Rights Reserved.


require('cutorch')

local withDevice = cutorch.withDevice

local gpu_local_copy_buffers = {}

--[[
`nn.AbstractParallel` is the base class for modules controlling
data/model-parallel behaviour in Torch.

The key concept is that data/model-parallelism _splits_ along a
dimension, and this class controls the distribution of input and
merging of output along this dimension.

To extend this class, override `_distributeInput` as appropriate.

See `nn.DataParallel` and `nn.ModelParallel` for examples of usage.
]]
local AbstractParallel, parent = torch.class('nn.AbstractParallel',
                                             'nn.Container')

function AbstractParallel:__init(dimension)
    if not dimension then
        error "must specify a dimension!"
    end
    parent.__init(self)
    self.modules = {}
    self.gpu_assignments = {}
    self.size = torch.LongStorage()
    self.dimension = dimension
    self.container_gpuid = cutorch.getDevice()

    self.input_gpu = {}  -- inputs for each gpu
    self.gradOutput_gpu = {} -- inputs for each gpu
    self.gradInput_gpu = {} -- gradInput for each gpu
end

function AbstractParallel:_freeCaches()
    self.input_gpu = {}
    self.gradOutput_gpu = {}
    self.gradInput_gpu = {}
end

--[[
This function yields the GPU id for the module to be added.

It can be used for load balancing. It assumes all GPUs are available.
]]
function AbstractParallel:nextGPU()
    local gpuid = #self.gpu_assignments % cutorch.getDeviceCount() + 1
    return gpuid
end

function AbstractParallel._getBuffer()
    local device = cutorch.getDevice()
    if not gpu_local_copy_buffers[device] then
        gpu_local_copy_buffers[device] = torch.CudaTensor()
    end
    return gpu_local_copy_buffers[device]
end


function AbstractParallel:add(module, gpuid)
    table.insert(self.modules, module)
    local gpuid = gpuid or self:nextGPU()
    table.insert(self.gpu_assignments, gpuid)
    return self
end

function AbstractParallel:get(index)
    return self.modules[index]
end

--[[
Asynchronous copy from dest to source.

Use with caution; there needs to be some sort of external synchronization to
prevent source from being modified after this copy is enqueued.
]]
function AbstractParallel:gpuSend(dest, source)
    assert(torch.typename(dest) == 'torch.CudaTensor')
    assert(torch.typename(source) == 'torch.CudaTensor')
    local dest_gpuid = dest:getDevice()
    local source_gpuid = source:getDevice()
    if source_gpuid == dest_gpuid then
        -- if both tensors are on the same gpu normal copy works
        withDevice(dest_gpuid, function()
            dest:copy(source)
        end)
        return
    end
    -- if both tensors are contiguous copy across gpus works
    if source:isContiguous() and dest:isContiguous() then
        withDevice(dest_gpuid, function() dest:copy(source) end)
        return
    end

    local tmp_source = source
    if not source:isContiguous() then
        withDevice(source_gpuid, function()
                       self:_getBuffer():resizeAs(source)
                       self:_getBuffer():copy(source)
                       tmp_source = self:_getBuffer()
        end)
    end

    withDevice(dest_gpuid, function()
                   -- if destination is not contiguous copy across gpus does not
                   -- work; we need to first copy the source to the dest gpu
                   if not dest:isContiguous() then
                      self:_getBuffer():resizeAs(tmp_source)
                      self:_getBuffer():copy(tmp_source)
                      dest:copy(self:_getBuffer())
                   else
                      dest:copy(tmp_source)
                   end

    end)
end

function AbstractParallel:updateOutput(input)
    local container_gpuid = cutorch.getDevice()
    local outs = {}

    -- distribute the input to GPUs
    self:_distributeInput(input)

    -- update output for each module.
    for i, module in ipairs(self.modules) do
        local gpuid = self.gpu_assignments[i]
        withDevice(gpuid, function()
                       assert(self.input_gpu[gpuid]:getDevice() ==
                                  self.gpu_assignments[gpuid])
                       outs[i] = module:updateOutput(self.input_gpu[gpuid])
        end)
    end

    -- find the size of the merged output.
    assert(container_gpuid == self.gpu_assignments[1])
    assert(outs[1].getDevice and
           (outs[1]:getDevice() == 0 or
            outs[1]:getDevice() == container_gpuid))
    self.size:resize(outs[1]:dim()):copy(outs[1]:size())
    for i=2,#outs do
            self.size[self.dimension] =
                self.size[self.dimension] + outs[i]:size(self.dimension)
    end

    -- merge (concatenate) the outputs
    self.output:resize(self.size)
    local offset = 1
    for i=1,#outs do
        local outputDim = outs[i]:size(self.dimension)
        local output_narrowed =
            self.output:narrow(self.dimension, offset, outputDim)
        self:gpuSend(output_narrowed, outs[i])
        offset = offset + outputDim
    end

    return self.output
end

function AbstractParallel:_distributeGradOutput(_input, gradOutput)
    local container_gpuid = cutorch.getDevice()

    -- distribute gradOutput chunks to modules
    local offset = 1
    for i,module in ipairs(self.modules) do
        local gpuid = self.gpu_assignments[i]
        withDevice(gpuid, function()
            local currentOutput = module.output

            -- get the gradOutput chunk for this module
            local currentGradOutput =
            gradOutput:narrow(self.dimension, offset,
                              currentOutput:size(self.dimension))

            self.gradOutput_gpu[i] = self.gradOutput_gpu[i] or torch.CudaTensor()
            self.gradOutput_gpu[i]:resizeAs(currentGradOutput)
            if gpuid == container_gpuid then
                self.gradOutput_gpu[i]:copy(currentGradOutput)
            else
                -- copy gradoutput chunk to module's gpu
                self:gpuSend(self.gradOutput_gpu[i], currentGradOutput)
            end

            offset = offset + currentOutput:size(self.dimension)
        end)
    end
end

function AbstractParallel:updateGradInput(_input, gradOutput)
   error('Not implemented')
end

function AbstractParallel:_mixGrads()
end

function AbstractParallel:accGradParameters(_input, _gradOutput, scale)
    scale = scale or 1
    for i,module in ipairs(self.modules) do
        local gpuid = self.gpu_assignments[i]
        withDevice(gpuid, function()
            module:accGradParameters(self.input_gpu[gpuid],
            self.gradOutput_gpu[i],
            scale)
        end)
    end
    -- Combine gradients for data parallel models
    self:_mixGrads()
end

function AbstractParallel:accUpdateGradParameters(_input, _gradOutput, lr)
    for i,module in ipairs(self.modules) do
        local gpuid = self.gpu_assignments[i]
        withDevice(gpuid, function()
            module:accUpdateGradParameters(self.input_gpu[gpuid], self.gradOutput_gpu[i], lr)
        end)
    end
end

function AbstractParallel:zeroGradParameters()
    for i,module in ipairs(self.modules) do
        withDevice(self.gpu_assignments[i], function()
            module:zeroGradParameters()
        end)
    end
end

function AbstractParallel:updateParameters(learningRate)
    for i,module in ipairs(self.modules) do
        withDevice(self.gpu_assignments[i], function()
            module:updateParameters(learningRate)
        end)
    end
end

function AbstractParallel:share(mlp,...)
    error("Share is not supported for the AbstractParallel layer.")
end

function AbstractParallel:clone()
    local clone = parent.clone(self)
    clone:cuda()
    return clone
end

function AbstractParallel:reset(stdv)
    for i,module in ipairs(self.modules) do
        withDevice(self.gpu_assignments[i], function()
            self.modules[i]:reset(stdv)
        end)
    end
end

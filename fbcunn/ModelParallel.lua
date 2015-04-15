-- Copyright 2004-present Facebook. All Rights Reserved.
require('cutorch')

local withDevice = cutorch.withDevice

--[[ `ModelParallel` copies inputs to all child modules, and runs
disjoint parts of the model on separate devices.

For example, consider a convolutional layer with a large number of
filter banks. ModelParallel will split the model along the given
`dimension` (e.g. 2 if we lay the input out as `BDWH`), copy the input
to each device, and then merge the outputs across the device.
]]
local ModelParallel, parent = torch.class('nn.ModelParallel',
                                          'nn.AbstractParallel')

function ModelParallel:__init(dimension)
    if not dimension then
        error "must specify a dimension!"
    end
    parent.__init(self, dimension)
    self.modules = {}
    self.gpu_assignments = {}
    self.size = torch.LongStorage()
    self.dimension = dimension
    self.container_gpuid = cutorch.getDevice()
end

function ModelParallel:_freeCaches()
    self.input_gpu = {}
    self.gradOutput_gpu = {}
end

function ModelParallel:nextGPU()
    -- This function yields the GPU id for the module to be added.
    -- It can be used for load balancing. It assumes all GPUs are available.
    local gpuid = #self.gpu_assignments % cutorch.getDeviceCount() + 1
    return gpuid
end

function ModelParallel:add(module, gpuid)
    table.insert(self.modules, module)
    local gpuid = gpuid or self:nextGPU()
    table.insert(self.gpu_assignments, gpuid)
    return self
end

function ModelParallel:get(index)
    return self.modules[index]
end

function ModelParallel:_distributeInput(input)
    local container_gpuid = cutorch.getDevice()
    -- For simplicity, we require the container starts at the first
    -- gpuid. This is trivial to get around via re-enumerating GPU
    -- ids, but the invariant holds for the current ImageNet
    -- architecture.
    assert(container_gpuid == 1)
    for i, _ in ipairs(self.modules) do

        -- Model our reduce tree as a binary heap. Thus, iterating
        -- over modules indices in increasing order corresponds to a
        -- BFS traversal of our binary heap. We issue copies from a
        -- parent node to a child node, and rely on CUDA's guarantee
        -- that asynchonous memcpy's WRT the NULL stream are totally
        -- ordered.
        -- See http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#multi-device-system
        -- In the four GPU case, we dispatch copies from GPU 1 to GPU
        -- 2, GPU 1 to GPU 3, and GPU 2 to GPU 4.

        -- For more than 4 GPUs, this is asymptotically less efficient
        -- than laying this out as a binomial heap (thanks @tconerly
        -- for pointing that out).

        local gpuid = self.gpu_assignments[i]
        local source = input
        if i > 1 then
            -- If i == 1, we're at the root of our binary heap.
            -- Otherwise, find the parent in the binary tree.
            local source_gpuid = self.gpu_assignments[math.floor(i / 2)]
            source = self.input_gpu[source_gpuid]
        end

        withDevice(gpuid, function()
                       -- move input to gpu if required
                       if gpuid == container_gpuid then
                           self.input_gpu[gpuid] = source
                       else
                           if not self.input_gpu[gpuid] then
                               self.input_gpu[gpuid] = torch.CudaTensor()
                           end

                           self.input_gpu[gpuid]:resizeAs(input)
                           self:gpuSend(self.input_gpu[gpuid], source)
                       end
        end)
end
end

function ModelParallel:name()
    return 'ModelParallel'
end

function ModelParallel:__tostring__()
    local tab = '  '
    local line = '\n'
    local next = '  |`-> '
    local ext = '  |    '
    local last = '   ... -> '
    local str = self:name()
    str = str .. ' {' .. line .. tab .. 'input'
    for i=1,#self.modules do
        local mod_str = tostring(self.modules[i]):gsub(line, line .. tab .. ext)
        str = string.format('%s%s%s%s(%d) [gpu:%d] %s',
                            str, line, tab, next,
                            i, self.gpu_assignments[i],
                            mod_str)
    end
    str = str .. line .. tab .. last .. 'output'
    str = str .. line .. '}'
    return str
end

-- Iterates over all key/value pairs in a table, in reverse order.
-- Equivalent to ipairs(t) in reverse.  See
-- http://lua-users.org/wiki/IteratorsTutorial for more detail.
local function ripairs(t)
    local max = 1
    while t[max] ~= nil do
        max = max + 1
    end

    local function ripairs_it(t, i)
        i = i-1
        local v = t[i]
        if v ~= nil then
            return i,v
        else
            return nil
        end
    end
    return ripairs_it, t, max
end

function ModelParallel:updateGradInput(_input, gradOutput)
   self:_distributeGradOutput(_input, gradOutput)

   -- update gradInput for each module
    for i,module in ipairs(self.modules) do
        local gpuid = self.gpu_assignments[i]
        withDevice(gpuid, function()
            module:updateGradInput(self.input_gpu[gpuid],
                                   self.gradOutput_gpu[i])
        end)
    end

    if not self.gradInput then return end -- if gradInput is nil, do nothing
    self.gradInput:resizeAs(self.input_gpu[self.container_gpuid])

    -- add gradInputs
    for i, module in ripairs(self.modules) do
        if module.gradInput then
            if i == 1 then
                self.gradInput:copy(module.gradInput)
                return self.gradInput
            end

            local parent_module_idx = math.floor(i / 2)
            local parent_gpuid = self.gpu_assignments[parent_module_idx]
            withDevice(parent_gpuid, function()
                           if not self.gradInput_gpu[i] then
                               self.gradInput_gpu[i] = torch.CudaTensor()
                           end

                           self.gradInput_gpu[i]:resizeAs(module.gradInput)
                           self:gpuSend(self.gradInput_gpu[i], module.gradInput)
                           self.modules[parent_module_idx].gradInput:add(
                               self.gradInput_gpu[i])
            end)
        end
    end

    return self.gradInput
end

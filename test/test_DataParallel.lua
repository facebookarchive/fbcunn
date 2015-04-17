local fboptim = require('fboptim')
-- Copyright 2004-present Facebook. All Rights Reserved.

local dprintL = (require 'fb.util.dbg').new('parallel')
local dprint = function(...)
    return dprintL(1, ...)
end
require 'optim'
require 'fbcunn'
print 'Requiring cunn. This will take a while. Talk amongst yourselves.'
require 'cunn'

-- Hyper-params. We're targeting a toy problem that computes
-- some function of its inputs.
local inputWidth = 32
local hiddenWidth = 512
local nHidden = 2
local outputWidth = 1
local numGPUs = cutorch.getDeviceCount()

local function targetFunction(x)
    -- admittedly tough for us to learn, but hey.
    local retval = torch.Tensor(outputWidth)
    local sum = x:sum()
    retval[1] = math.sin(sum)
    return retval
end

local function genInput()
   return torch.randn(inputWidth)
end

local function genWideInput()
   return torch.randn(inputWidth * numGPUs)
end

local function getNarrowedInputRange(i)
    assert(type(i) == 'number')
    local rangeStart = 1 + ((i - 1) * inputWidth)
    local rangeEnd = rangeStart + (inputWidth) - 1
    return rangeStart, rangeEnd
end

local function getNarrowedInput(input, i)
    assert(torch.typename(input))
    assert(type(i) == 'number')
    return input[{ {getNarrowedInputRange(i)} }]
end

local function genWideExample()
    local samp = genWideInput()
    local retval = torch.Tensor(outputWidth * numGPUs)
    for i = 1,numGPUs do
        retval[i] = targetFunction(getNarrowedInput(samp, i))
    end
    return samp:cuda(), retval:cuda()
end

local function simpleModel()
    local seq = nn.Sequential()
    local pred = inputWidth
    for i = 1,nHidden do
       seq:add(nn.Linear(pred, hiddenWidth))
       seq:add(nn.Tanh())
       pred = hiddenWidth
    end
    seq:add(nn.Linear(hiddenWidth, outputWidth))
    seq:add(nn.Tanh())
    return seq
end

local function tensorsAreProbablySimilar(l, r, epsilon)
    epsilon = epsilon or 0.00001
    return math.abs(l:norm() - r:norm()) < epsilon
end

-- Set up models on each GPU.
local dp = nn.DataParallel(1)
local simpleModels = {}
for i = 1,numGPUs do
    if i == 1 then
        simpleModels[i] = simpleModel()
    else
        simpleModels[i] = simpleModels[1]:clone()
    end
    dp:add(simpleModels[i])
end

-- CPU models to cross-validate
local cpuModels = {}
local function syncCPUModels()
    for i = 1,numGPUs do
        cpuModels[i] = simpleModels[i]:clone()
        cpuModels[i] = cpuModels[i]:double()
    end
end
syncCPUModels()

-- Check an input/output pair against the CPU models
local function checkWideResult(inputs, outputs)
    local function checkOneResult(input, modIdx, expectedOutput)
        input = input:double() -- de-cudify
        assert(tensorsAreProbablySimilar(cpuModels[modIdx]:forward(input),
                                         expectedOutput))
    end
    for j = 1, numGPUs do
        checkOneResult(getNarrowedInput(inputs, j), j, outputs[{ {j} }])
    end
end

local function checkCPUModelsAreEquivalent()
    syncCPUModels()
    local input = genInput()
    local out = cpuModels[1]:forward(input)
    for j = 2, numGPUs do
        assert(tensorsAreProbablySimilar(out, cpuModels[j]:forward(input)))
    end
end
checkCPUModelsAreEquivalent()

dp:cuda()

-- Make sure forward produces same results as an individual copy
print('forward test {')
for i=1, 10 do
    local inputs, targets = genWideExample()
    dprint{ inputs, targets }
    local outputs = dp:forward(inputs)
    syncCPUModels()
    checkWideResult(inputs, outputs)
end
print('} forward test done')

print('optim test {')
local optimState = {
    learningRate = 1e-1,
    weightDecay = 1e-4,
    momentum = 0.9,
    learningRateDecay = 1e-7
}

local timer = torch.Timer()
local opt = nn.Optim(dp, optimState)
local criterion = nn.MSECriterion():cuda()

local num_iteration = 10
timer:reset()
for i=1, num_iteration do
    local inputs, targets = genWideExample()
    local outputs = dp:forward(inputs)
    syncCPUModels()
    checkWideResult(inputs, outputs)
    opt:optimize(fboptim.sgd, inputs, targets, criterion)
    local out = dp:forward(inputs)
    local err = criterion:forward(out, targets)
    print(i, err)
end
print(string.format("Total time spent = %f", timer:time().real / num_iteration))
checkCPUModelsAreEquivalent()
print('} optim test done ')

-- Check only the speed for forward/backward.
timer:reset();
for i=1, num_iteration do
    local inputs, targets = genWideExample()
    dp:forward(inputs)
    opt:optimize(fboptim.sgd, inputs, targets, criterion)
end
print(string.format(
    "Speedtest: Total time spent = %f",
        timer:time().real / num_iteration));

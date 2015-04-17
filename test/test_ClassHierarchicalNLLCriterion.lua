-- Copyright 2004-present Facebook. All Rights Reserved.
-- Train simple 1 hidden layer neural net on MNIST using the hierarchical
-- softmax layer (with the option to test also the regular flat softmax as well)
require('fb.luaunit')
local torch = require('fbtorch')
require('nn')
require('fbcunn')
require('os')

torch.setdefaulttensortype('torch.DoubleTensor')
torch.manualSeed(111)
local noTrainingDemo = true
-- we can't use the tester from Torch because this is a Criterion with
-- parameters.
local debugger = require('fb.debugger')
local basic = true

local dimInput = 15
local epsilon = 1e-6
local tolRatio = 1e-3
local tolDiff = 4e-4
local map = torch.Tensor{{1, 1},
                  {2, 1},
                  {3, 1},
                  {4, 1},
                  {5, 1},
                  {4, 2},
                  {3, 3},
                  {2, 2},
                  {3, 2},
                  {5, 2}}
local clusterCounts = torch.Tensor{1, 2, 3, 2, 2}

-- basic setup
if basic then
    map = torch.Tensor{{1,1},{1,2},{1,3},{1,4},{1,5},{1,6},{1,7},{1,8},{1,9},
                      {1,10}}
    clusterCounts = torch.Tensor{10}
    -- When testing the flat softmax, compare with the default implementation.
    modelDefault = nn.Sequential()
    modelDefault:add(nn.Reshape(dimInput))
    modelDefault:add(nn.Linear(dimInput, clusterCounts[1]))
    modelDefault:add(nn.LogSoftMax())
    criterionDefault = nn.ClassNLLCriterion()
end

local criterion = nn.ClassHierarchicalNLLCriterion(map,
    clusterCounts, dimInput)

-- testing the gradient input
local passTestInput = true
local passAllTests = true
for label = 1, 10 do
    local randInput = torch.randn(dimInput)
    local inputPlus = torch.zeros(dimInput)
    local inputMinus = torch.zeros(dimInput)

    local loss = criterion:forward(randInput, label)
    local grad = criterion:updateGradInput(randInput, label)
    if basic then
       -- copy parameters
       modelDefault.modules[2].weight:copy(criterion.classMatrix)
       modelDefault.modules[2].bias:copy(criterion.classBias)
       local lossDefault = criterionDefault:forward(
          modelDefault:forward(randInput), label)
       modelDefault:zeroGradParameters()
       criterionDefault:backward(modelDefault.output, label)
       gradDefault = modelDefault:backward(randInput,
                                           criterionDefault.gradInput)
    end

    for i = 1, dimInput do
        inputPlus:copy(randInput)
        inputMinus:copy(randInput)

        inputPlus[i] = inputPlus[i] + epsilon
        inputMinus[i] = inputMinus[i] - epsilon
        assert((randInput - (inputPlus + inputMinus) / 2):sum() == 0)

        local lossPlus = criterion:forward(inputPlus, label)
        local lossMinus = criterion:forward(inputMinus, label)
        local estimate = (lossPlus - lossMinus) / (2 * epsilon)
        passTestInput = (grad[i] > 1) and
           (math.abs(estimate / grad[i] - 1) < tolRatio) or
           (math.abs(estimate - grad[i]) < tolDiff)
        passAllTests = passAllTests and passTestInput
        if not passTestInput then
            print('The gradient w.r.t. the input is different from the'
                .. ' numerical estimate. BPROP value '
                .. grad[i] .. ' , numerical estimate ' .. estimate
                .. ', difference ' .. math.abs(estimate - grad[i])
            )
        end
        if basic then
           assert(math.abs(grad[i] - gradDefault[i]) < 1e-16)
        end
    end
end

-- class and cluster biases
for _, x in pairs{{criterion.clusterBias, criterion.clusterBiasDx},
                  {criterion.classBias, criterion.classBiasDx}} do
    local param = x[1]
    local pGrad = x[2]
    for label = 1, 10 do
        local dim = param:size(1)

        local randParam = torch.randn(dim)
        local paramPlus = torch.zeros(dim)
        local paramMinus = torch.zeros(dim)

        local randInput = torch.randn(dimInput)
        param:copy(randParam)
        criterion:zeroGradParameters()
        local loss = criterion:forward(randInput, label)
        criterion:updateGradInput(randInput, label)

        if basic then
           modelDefault.modules[2].weight:copy(criterion.classMatrix)
           modelDefault.modules[2].bias:copy(criterion.classBias)
           criterionDefault:forward(
              modelDefault:forward(randInput), label)
           modelDefault:zeroGradParameters()
           criterionDefault:backward(modelDefault.output, label)
           modelDefault:backward(randInput, criterionDefault.gradInput)
        end

        for i = 1, dim do
            paramPlus:copy(randParam)
            paramMinus:copy(randParam)

            paramPlus[i] = paramPlus[i] + epsilon
            paramMinus[i] = paramMinus[i] - epsilon

            assert((randParam - (paramPlus + paramMinus) / 2):sum() == 0)

            param:copy(paramPlus)
            local lossPlus = criterion:forward(randInput, label)
            param:copy(paramMinus)
            local lossMinus = criterion:forward(randInput, label)
            local estimate = (lossPlus - lossMinus) / (2 * epsilon)

            passTestInput = (pGrad[i] > 1) and
               (math.abs(estimate / pGrad[i] - 1) < tolRatio) or
               (math.abs(estimate - pGrad[i]) < tolDiff)
            passAllTests = passAllTests and passTestInput
            if not passTestInput then
               print('Label ' .. label .. ', id ' .. i
                        .. '. The gradient w.r.t. the biases is different from'
                        .. ' the numerical estimate. BPROP value '
                        .. pGrad[i] .. ' , numerical estimate ' .. estimate
                        .. ', difference ' .. math.abs(estimate - pGrad[i])
               )
            end
            if basic then
               assert(math.abs(
                         criterion.classBiasDx[i] -
                            modelDefault.modules[2].gradBias[i]) <
                         1e-16)
            end
        end
    end
end

-- class and cluster matrices
for _, x in pairs{{criterion.clusterMatrix, criterion.clusterMatrixDx},
                  {criterion.classMatrix, criterion.classMatrixDx}} do
    local param = x[1]
    local pGrad = x[2]
    for label = 1, 10 do
        local dim = param:size()

        local randParam = torch.randn(dim)
        local paramPlus = torch.zeros(dim)
        local paramMinus = torch.zeros(dim)

        local randInput = torch.randn(dimInput)
        param:copy(randParam)
        criterion:zeroGradParameters()
        local loss = criterion:forward(randInput, label)
        criterion:updateGradInput(randInput, label)

        if basic then
           modelDefault.modules[2].weight:copy(criterion.classMatrix)
           modelDefault.modules[2].bias:copy(criterion.classBias)
           criterionDefault:forward(
              modelDefault:forward(randInput), label)
           modelDefault:zeroGradParameters()
           criterionDefault:backward(modelDefault.output, label)
           modelDefault:backward(randInput, criterionDefault.gradInput)
        end

        for i = 1, dim[1] do
          for j = 1, dim[2] do
            paramPlus:copy(randParam)
            paramMinus:copy(randParam)

            paramPlus[i][j] = paramPlus[i][j] + epsilon
            paramMinus[i][j] = paramMinus[i][j] - epsilon

            assert((randParam - (paramPlus + paramMinus) / 2):sum() == 0)

            param:copy(paramPlus)
            local lossPlus = criterion:forward(randInput, label)
            param:copy(paramMinus)
            local lossMinus = criterion:forward(randInput, label)
            local estimate = (lossPlus - lossMinus) / (2 * epsilon)

            passTestInput = (pGrad[i][j] > 1) and
               (math.abs(estimate / pGrad[i][j] - 1) < tolRatio) or
               (math.abs(estimate - pGrad[i][j]) < tolDiff)
            passAllTests = passAllTests and passTestInput
            if not passTestInput then
               print('Label ' .. label .. ', id ' .. i .. ' ' .. j
                        .. '. The gradient w.r.t. the biases is different from'
                        .. ' the numerical estimate. BPROP value '
                        .. pGrad[i][j] .. ' , numerical estimate ' .. estimate
                        .. ', difference ' .. math.abs(estimate - pGrad[i][j])
               )
            end
            if basic then
               assert(math.abs(
                         criterion.classMatrixDx[i][j] -
                            modelDefault.modules[2].gradWeight[i][j]) <
                         1e-16)
            end
          end
        end
    end
end
assert(passAllTests)

if noTrainingDemo then
   os.exit()
end


-- @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
print '==> processing options'
local cmd = torch.CmdLine()
cmd:text('Options:')
cmd:option('-learningRate', 1e-2, 'learning rate at t=0')
cmd:option('-numEpochs', 2, 'number of epochs for SGD')
cmd:option('-numHiddens', 256, 'number of hidden units')
cmd:option('-baseline', 0, '0 HSM, 1 flat softmax')
cmd:option('-oneCluster', 0, '0 use 5 clusters, 1 use one cluster')
cmd:text()
local opt = cmd:parse(arg or {})
print(opt)
print '==> loading and preprocessing data'
local train_file = '/gpustorage/ai-group/datasets/mnist/train_32x32.t7'
local test_file = '/gpustorage/ai-group/datasets/mnist/test_32x32.t7'

local data = torch.load(train_file, 'ascii')
local trsize = data.data:size()[1]
local numChannels = data.data:size()[2]
local numpix = data.data:size()[3] * data.data:size()[4]

local trainData = {
    data = data.data:resize(trsize,numChannels*numpix):float() / 255.0,
    labels = data.labels,
    size = function() return trsize end
}

data = torch.load(test_file, 'ascii')
local tesize = data.data:size()[1]

local testData = {
    data = data.data:resize(tesize,numChannels*numpix):float() / 255.0,
    labels = data.labels,
    size = function() return tesize end
}

-- @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
print '==> building the model'
local ninputs = numChannels * numpix
local nhiddens = opt.numHiddens
local noutputs = data.labels:max()
local model = nn.Sequential()
model:add(nn.Reshape(ninputs))
model:add(nn.Linear(ninputs,nhiddens))
model:add(nn.Threshold(0.0, 0.0))
local mapping, clusterCounts
if opt.oneCluster == 0 then
   -- Note labels are shifted by 1 (e.g., "0" has label 1)
   -- Clusters: {0}, {1,7}, {2,8,6}, {3,5}, {4,9}
   mapping = torch.Tensor{{1, 1},
                          {2, 1},
                          {3, 1},
                          {4, 1},
                          {5, 1},
                          {4, 2},
                          {3, 3},
                          {2, 2},
                          {3, 2},
                          {5, 2}}
   clusterCounts = torch.Tensor{1, 2, 3, 2, 2}
else -- just one cluster (equivalent to flat softmax)
   mapping = torch.Tensor{{1,1},{1,2},{1,3},{1,4},{1,5},{1,6},{1,7},{1,8},{1,9},
                          {1,10}}
   clusterCounts = torch.Tensor{10}
end
local criterion = nn.ClassHierarchicalNLLCriterion(mapping, clusterCounts,
                                                   nhiddens)
if opt.baseline == 1 then
   model:add(nn.Linear(nhiddens, 10))
   model:add(nn.LogSoftMax())
   criterion = nn.ClassNLLCriterion()
   model.modules[2].bias:fill(0.0)
   model.modules[4].bias:fill(0.0)
end
print(model, criterion)

local input = torch.Tensor(trainData.data:size()[2])
local target = 0
-- Train for one epoch.
function train()
    -- epoch tracker
    epoch = epoch or 1
    -- local vars
    local time = sys.clock()
    -- shuffle at each epoch
    shuffle = torch.randperm(trsize)
    local loss = 0
    print("Epoch # " .. epoch)

    for ss = 1, trainData:size() do
       input:copy(trainData.data[shuffle[ss]])
       target = trainData.labels[shuffle[ss]]
       loss = loss + criterion:forward(model:forward(input), target)
       model:zeroGradParameters()
       if opt.baseline == 0 then
          criterion:zeroGradParameters()
       end
       criterion:backward(model.output, target)
       model:backward(input, criterion.gradInput)
       model:updateParameters(opt.learningRate)
       if opt.baseline == 0 then
          criterion:updateParameters(opt.learningRate)
       end
    end
    -- time taken
    time = sys.clock() - time
    print("Loss " .. loss / trainData:size() .. ", time " .. time ..
             's')
    epoch = epoch + 1
end

function predict(d)
   local err = 0
   for ss = 1, d:size() do
      input:copy(d.data[ss])
      target = d.labels[ss]
      local prediction = {}
      if opt.baseline == 0 then -- hierarchical softmax
         prediction = criterion:infer(model:forward(input))
         if prediction[1] ~= mapping[target][1] or
           prediction[2] ~= mapping[target][2] then
             err = err + 1
         end
      else -- flat softmax baseline
         local output = model:forward(input)
         local val, indx = torch.max(output, 1)
         if indx[1] ~= target then
            err = err + 1
         end
      end
    end
   return err / d:size()
end

function compute_logprob_rankerr(d)
   local logProb = 0
   local rankErr = 0
   for ss = 1, d:size() do
      input:copy(d.data[ss])
      target = d.labels[ss]
      local prediction = {}
      local currLogProb, currRankErr = criterion:eval(
         model:forward(input), target)
      logProb = logProb + currLogProb
      rankErr = rankErr + currRankErr
    end
   logProb = logProb / d:size()
   rankErr = rankErr / d:size()
   print('LogProb ' .. logProb .. ', rank error ' .. rankErr)
   return logProb, rankErr
end

function test()
   print('Training error ' .. predict(trainData))
   print('Test error ' .. predict(testData))
end

for cnt = 1, opt.numEpochs do
    train()
    test()
end
if opt.baseline == 0 then
   local logp, ranke = compute_logprob_rankerr(testData)
   assert(math.abs(logp + 0.040576076568463) < 0.001)
   assert(math.abs(ranke - 0.0046) < 0.001)
end

-- Copyright 2004-present Facebook. All Rights Reserved.
-- Author: Michael Mathieu <myrhev@fb.com>

require 'fbcunn'
require 'fbnn'
require 'math'
require 'nn'
require 'os'
require 'sys'

torch.setdefaulttensortype('torch.DoubleTensor')
torch.manualSeed(111)
local debugger = require 'fb.debugger'

local numIters = 10000
local dimInput = 1000
local numClasses = 10000
-- map1 is single cluster softmax
local map1 = torch.ones(numClasses, 2)
for c = 1, numClasses do
   map1[c][1] = c
end
local clusterCounts1 = torch.ones(numClasses)
-- map2 is a hierarhcical softmax, 10 clusters - each with 100 classes
local map2 = torch.ones(numClasses, 2)
local numClusters = 100
assert(numClusters * (numClasses / numClusters) == numClasses)
for c = 1, numClasses do
   map2[c][1] = math.floor((c - 1) / (numClasses / numClusters)) + 1
   map2[c][2] = ((c - 1) % (numClasses / numClusters)) + 1
end
local clusterCounts2 = torch.ones(numClusters) * (numClasses / numClusters)

local modelDefault = nn.Sequential()
modelDefault:add(nn.Reshape(dimInput))
modelDefault:add(nn.Linear(dimInput, numClasses))
modelDefault:add(nn.LogSoftMax())
local criterionDefault = nn.ClassNLLCriterion()

local criterionMrFlat = nn.ClassHierarchicalNLLCriterion(
   map1, clusterCounts1, dimInput)

local criterionMrHier = nn.ClassHierarchicalNLLCriterion(
   map2, clusterCounts2, dimInput)

local criterionMmFlat = nn.HSM(map1, dimInput)

local criterionMmHier = nn.HSM(map2, dimInput)

local randInput = torch.randn(dimInput)
local batchSize = 128
local randInputBatch = torch.randn(batchSize, dimInput)
local targetBatch = torch.LongTensor(batchSize)
for i = 1, batchSize do
   targetBatch[i] = torch.random(numClasses)
end
local loss
print('Done with initialization')

print('FPROP Batch size = 1')
-- BATCH SIZE 1
local timeAll = torch.zeros(5)

function timeitFPROP(obj, str, use_tensor)
   local time = sys.clock()
   for iter = 1, numIters do
      loss = obj:forward(randInput, use_tensor
                            and torch.LongTensor{((iter-1) % numClasses) + 1}
                            or ((iter-1) % numClasses) + 1)
   end
   time = sys.clock() - time
   print('Done ' .. str .. ' in ' .. time .. 's')
   return time
end

local time = sys.clock()
for iter = 1, numIters do
   loss = criterionDefault:forward(
      modelDefault:forward(randInput), ((iter-1) % numClasses) + 1)
end
timeAll[1] = sys.clock() - time
print('Done default flat in ' .. timeAll[1] .. 's')

timeAll[2] = timeitFPROP(criterionMrFlat, 'MR flat')
timeAll[3] = timeitFPROP(criterionMmFlat, 'MM flat', true)
timeAll[4] = timeitFPROP(criterionMrHier, 'MR hier')
timeAll[5] = timeitFPROP(criterionMmHier, 'MM hier', true)

timeAllBatch = torch.zeros(3)
print('FPROP Batch size = ' .. batchSize)
local time = sys.clock()
for iter = 1, numIters do
   loss = criterionDefault:forward(
      modelDefault:forward(randInputBatch), targetBatch)
end
timeAllBatch[1] = sys.clock() - time
print('Done default flat batch=' .. batchSize .. ' in ' ..
         timeAllBatch[1] .. 's')

function timeitBatchFPROP(obj, str)
   local time = sys.clock()
   for iter = 1, numIters do
      loss = obj:forward(randInputBatch, targetBatch)
   end
   time = sys.clock() - time
   print('Done ' .. str .. ' in ' .. time .. 's')
   return time
end

timeAllBatch[2] = timeitBatchFPROP(
   criterionMmFlat, 'MM flat batch=' .. batchSize)
timeAllBatch[3] = timeitBatchFPROP(
   criterionMmHier, 'MM hier batch=' .. batchSize)


------------- TEST BPROP -------------
print('-------------------------------------')
print('BPROP Batch size = 1')
numIters = 100
-- BATCH SIZE 1
local timeAll = torch.zeros(5)

function timeitBPROP(obj, str, use_tensor)
   local time = sys.clock()
   for iter = 1, numIters do
      local tt = use_tensor
         and torch.LongTensor{((iter-1) % numClasses) + 1}
         or ((iter-1) % numClasses) + 1
      loss = obj:forward(randInput, tt)
      if tostring(obj) == 'nn.HSM' then
         obj:zeroGradParametersClass(randInput, tt)
      else
         obj:zeroGradParametersClass(tt)
      end
      obj:backward(randInput, tt)
   end
   time = sys.clock() - time
   print('Done ' .. str .. ' in ' .. time .. 's')
   return time
end

local time = sys.clock()
for iter = 1, numIters do
   loss = criterionDefault:forward(
      modelDefault:forward(randInput), ((iter-1) % numClasses) + 1)
   modelDefault:zeroGradParameters()
   criterionDefault:backward(modelDefault.output, ((iter-1) % numClasses) + 1)
   modelDefault:backward(randInput, criterionDefault.gradInput)
end
timeAll[1] = sys.clock() - time
print('Done default flat in ' .. timeAll[1] .. 's')
timeAll[2] = timeitBPROP(criterionMrFlat, 'MR flat')
timeAll[3] = timeitBPROP(criterionMmFlat, 'MM flat', true)
timeAll[4] = timeitBPROP(criterionMrHier, 'MR hier')
timeAll[5] = timeitBPROP(criterionMmHier, 'MM hier', true)

timeAllBatch = torch.zeros(3)
print('BPROP Batch size = ' .. batchSize)
local time = sys.clock()
for iter = 1, numIters do
   loss = criterionDefault:forward(
      modelDefault:forward(randInputBatch), targetBatch)
   modelDefault:zeroGradParameters()
   criterionDefault:backward(modelDefault.output, targetBatch)
   modelDefault:backward(randInputBatch, criterionDefault.gradInput)
end
timeAllBatch[1] = sys.clock() - time
print('Done default flat batch=' .. batchSize .. ' in ' ..
         timeAllBatch[1] .. 's')

function timeitBatchBPROP(obj, str)
   local time = sys.clock()
   for iter = 1, numIters do
      loss = obj:forward(randInputBatch, targetBatch)
      if tostring(obj) == 'nn.HSM' then
         obj:zeroGradParametersClass(randInputBatch, targetBatch)
      else
         obj:zeroGradParametersClass(targetBatch)
      end
      obj:backward(randInputBatch, targetBatch)
   end
   time = sys.clock() - time
   print('Done ' .. str .. ' in ' .. time .. 's')
   return time
end

timeAllBatch[2] = timeitBatchBPROP(
   criterionMmFlat, 'MM flat batch=' .. batchSize)
timeAllBatch[3] = timeitBatchBPROP(
   criterionMmHier, 'MM hier batch=' .. batchSize)

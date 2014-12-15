-- Copyright 2004-present Facebook. All Rights Reserved.

--[[
Hierarchical softmax classifier with two levels and arbitrary clusters.

Note:
This criterion does include the lower layer parameters
(this is more `Linear` + `ClassNLLCriterion`, but hierarchical).
Also, this layer does not support the use of mini-batches
(only 1 sample at the time).
]]
local ClassHierarchicalNLLCriterion, parent = torch.class(
   'nn.ClassHierarchicalNLLCriterion', 'nn.Criterion')

--[[
Parameters:

* `mapping` is a tensor with as many elements as classes.
   `mapping[i][1]` stores the cluster id, and `mapping[i][2]` the class id within
   that cluster of the ${i}$-th class.
* `clusterCounts` is a vector with as many entry as clusters.
   clusterCounts[i] stores the number of classes in the i-th cluster.
*  `inputSize` is the number of input features
]]
function ClassHierarchicalNLLCriterion:__init(mapping, clusterCounts, inputSize)
   parent.__init(self)
   local numClusters = clusterCounts:size(1)
   local numClasses = mapping:size(1)
   assert(numClasses == clusterCounts:sum())
   self.mapping = torch.Tensor(mapping)
   -- stores the start index of each cluster, useful to slice classMatrix
   self.startIndex = torch.ones(numClusters)
   for cc = 2, numClusters do
      self.startIndex[cc] = self.startIndex[cc - 1] + clusterCounts[cc - 1]
   end
   self.clusterCounts = torch.Tensor(clusterCounts)
   -- Parameters
   local stdev = 1./math.sqrt(inputSize)
   self.clusterMatrix = torch.randn(numClusters, inputSize) * stdev
   self.classMatrix = torch.randn(numClasses, inputSize) * stdev
   self.clusterBias = torch.zeros(numClusters)
   self.classBias = torch.zeros(numClasses)
   self.clusterMatrixDx = torch.zeros(numClusters, inputSize)
   self.classMatrixDx = torch.zeros(numClasses, inputSize)
   self.clusterBiasDx = torch.zeros(numClusters)
   self.classBiasDx = torch.zeros(numClasses)
   -- Log Losses for cluster and class prediction (the latter is shared across
   -- all clusters, since the grad. w.r.t. input is reshaped anyway).
   self.logSMCluster = nn.LogSoftMax()
   self.logSMClass = nn.LogSoftMax()
   self.logLossCluster = nn.ClassNLLCriterion()
   self.logLossClass = nn.ClassNLLCriterion()
   -- Buffers (values just before logSoftmax)
   self.outputCluster = torch.zeros(numClusters)
   self.outputClass = torch.zeros(numClasses)
   self.outputClusterDx = torch.zeros(numClusters)
   self.outputClassDx = torch.zeros(numClasses)
   -- Variables storing IDs for current sample
   self.clusterID = 0
   self.classID = 0
   self.startCluster = 0
   self.numClasses = 0
end

-- `target` is the class id
function ClassHierarchicalNLLCriterion:updateOutput(input, target)
   assert(input:dim() == 1) -- we do not support mini-batch training
   self.clusterID = self.mapping[target][1]
   self.classID = self.mapping[target][2]
   self.startCluster = self.startIndex[self.clusterID]
   self.numClasses = self.clusterCounts[self.clusterID]
   local loss = 0
   -- Prediction of cluster
   self.outputCluster:mv(self.clusterMatrix, input)
   self.outputCluster:add(self.clusterBias)
   loss = self.logLossCluster:forward(self.logSMCluster:forward(
                                         self.outputCluster), self.clusterID)
   -- Prediction of class within the cluster
   self.outputClass:narrow(1, self.startCluster, self.numClasses):mv(
      self.classMatrix:narrow(1, self.startCluster, self.numClasses), input)
   self.outputClass:narrow(1, self.startCluster, self.numClasses):add(
      self.classBias:narrow(1, self.startCluster, self.numClasses))
   loss = loss + self.logLossClass:forward(
      self.logSMClass:forward(
         self.outputClass:narrow(1, self.startCluster, self.numClasses)),
      self.classID)
   self.output = loss
   return self.output
end

function ClassHierarchicalNLLCriterion:zeroGradParameters()
   self.clusterBiasDx:zero()
   self.clusterMatrixDx:zero()
   self.classBiasDx:zero()
   self.classMatrixDx:zero()
end

function ClassHierarchicalNLLCriterion:zeroGradParametersCluster()
   self.clusterBiasDx:zero()
   self.clusterMatrixDx:zero()
end

function ClassHierarchicalNLLCriterion:zeroGradParametersClass(target)
   local clusterID = self.mapping[target][1]
   local startCluster = self.startIndex[clusterID]
   local numClasses = self.clusterCounts[clusterID]
   self.classMatrixDx:narrow(1, startCluster, numClasses):zero()
   self.classBiasDx:narrow(1, startCluster, numClasses):zero()
end

-- This computes derivatives w.r.t. input and parameters.
function ClassHierarchicalNLLCriterion:updateGradInput(input, target)
   assert(input:dim() == 1) -- we do not support mini-batch training
   self.gradInput:resizeAs(input)
   -- BPROP through the cluster prediction
   self.logLossCluster:updateGradInput(self.logSMCluster.output, self.clusterID)
   self.logSMCluster:updateGradInput(self.outputCluster,
                                     self.logLossCluster.gradInput)
   self.clusterBiasDx:add(self.logSMCluster.gradInput)
   self.gradInput:mv(self.clusterMatrix:t(), self.logSMCluster.gradInput)
   self.clusterMatrixDx:addr(self.logSMCluster.gradInput, input)
   -- BPROP through the cluster prediction
   self.logLossClass:updateGradInput(self.logSMClass.output, self.classID)
   self.logSMClass:updateGradInput(
      self.outputClass:narrow(1, self.startCluster, self.numClasses),
      self.logLossClass.gradInput)
   self.classBiasDx:narrow(1, self.startCluster, self.numClasses):add(
      self.logSMClass.gradInput)
   self.gradInput:addmv(
      self.classMatrix:narrow(1, self.startCluster, self.numClasses):t(),
      self.logSMClass.gradInput)
   self.classMatrixDx:narrow(1, self.startCluster, self.numClasses):addr(
      self.logSMClass.gradInput, input)
   return self.gradInput
end

-- Update parameters (only those that are used to process this sample).
function ClassHierarchicalNLLCriterion:updateParameters(learningRate)
   self.classMatrix:narrow(1, self.startCluster, self.numClasses):add(
         -learningRate,
      self.classMatrixDx:narrow(1, self.startCluster, self.numClasses))
   self.classBias:narrow(1, self.startCluster, self.numClasses):add(
         -learningRate,
      self.classBiasDx:narrow(1, self.startCluster, self.numClasses))
   self.clusterMatrix:add(-learningRate, self.clusterMatrixDx)
   self.clusterBias:add(-learningRate, self.clusterBiasDx)
end

-- input is a vector of probabilities (non-negative and sums to 1).
function sampleMultiNomial(input)
   local numVal = input:size(1)
   local rndu = (math.random(1000000) - 1) / (1000000 - 1)
   local tot = 0
   local cnt = 0
   for c = 1, numVal do
      cnt = cnt + 1
      tot = tot + input[c]
      if tot > rndu then break end
   end
   return cnt
end

-- Inference of the output (to be used at test time only)
-- If sampling flag is set to true, then the output label is sampled
-- o/w the most likely class is provided.
function ClassHierarchicalNLLCriterion:infer(input, sampling)
   assert(input:dim() == 1) -- we do not support mini-batch training
   -- Prediction of cluster
   self.outputCluster:mv(self.clusterMatrix, input)
   self.outputCluster:add(self.clusterBias)
   if sampling ~= nil and sampling == true then
      local prob = self.logSMCluster:forward(self.outputCluster)
      prob:exp()
      prob:div(prob:sum())
      self.clusterID = sampleMultiNomial(prob)
   else
      local val, indx = torch.max(self.outputCluster, 1)
      self.clusterID = indx[1]
   end
   self.startCluster = self.startIndex[self.clusterID]
   self.numClasses = self.clusterCounts[self.clusterID]
   -- Prediction of class within the cluster
   self.outputClass:narrow(1, self.startCluster, self.numClasses):mv(
      self.classMatrix:narrow(1, self.startCluster, self.numClasses), input)
   self.outputClass:narrow(1, self.startCluster, self.numClasses):add(
      self.classBias:narrow(1, self.startCluster, self.numClasses))
   if sampling ~= nil and sampling == true then
      local prob = self.logSMClass:forward(
         self.outputClass:narrow(1, self.startCluster, self.numClasses))
      prob:exp()
      prob:div(prob:sum())
      self.classID = sampleMultiNomial(prob)
   else
      local val, indx = torch.max(
         self.outputClass:narrow(1, self.startCluster, self.numClasses), 1)
      self.classID = indx[1]
   end
   return {self.clusterID, self.classID}
end

-- Given some label, it computes the logprob and the ranking error.
function ClassHierarchicalNLLCriterion:eval(input, target)
   self:updateOutput(input, target)
   self.logSMCluster.output:exp()
   self.logSMCluster.output:div(self.logSMCluster.output:sum())
   local logProb = math.log10(self.logSMCluster.output[self.clusterID])
   self.logSMClass.output:exp()
   self.logSMClass.output:div(self.logSMClass.output:sum())
   logProb = logProb + math.log10(self.logSMClass.output[self.classID])
   -- Estimate ranking error
   self.clusterID = math.random(self.logSMCluster.output:size(1))
   self.classID = math.random(self.clusterCounts[self.clusterID])
   local cnt = 1
   while self.clusterID == self.mapping[target][1] and
         self.classID == self.mapping[target][2] and cnt < 1000 do
            self.clusterID = math.random(self.logSMCluster.output:size(1))
            self.classID = math.random(self.clusterCounts[self.clusterID])
            cnt = cnt + 1
   end
   if cnt == 1000 then
      print('Warning ClassHierarchicalNLLCriterion:eval ' ..
               'I could not find a good negative sample!')
   end
   self.startCluster = self.startIndex[self.clusterID]
   self.numClasses = self.clusterCounts[self.clusterID]
   local logProbRandLabel =  math.log10(
      self.logSMCluster.output[self.clusterID])
   self.outputClass:narrow(1, self.startCluster, self.numClasses):mv(
      self.classMatrix:narrow(1, self.startCluster, self.numClasses), input)
   self.outputClass:narrow(1, self.startCluster, self.numClasses):add(
      self.classBias:narrow(1, self.startCluster, self.numClasses))
   self.logSMClass:forward(
      self.outputClass:narrow(1, self.startCluster, self.numClasses))
   self.logSMClass.output:exp()
   self.logSMClass.output:div(self.logSMClass.output:sum())
   logProbRandLabel = logProbRandLabel +
      math.log10(self.logSMClass.output[self.classID])
   local rankErr =  (logProb > logProbRandLabel) and 0 or 1
   return logProb, rankErr
end

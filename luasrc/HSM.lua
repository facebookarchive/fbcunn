-- Copyright 2004-present Facebook. All Rights Reserved.
-- Author: Michael Mathieu <myrhev@fb.com>


require 'math'
require 'nn'

-- Hierarchical soft max with minibatches.
local HSM, parent =
    torch.class('nn.HSM', 'nn.Criterion')

--[[
Parameters:
* `mapping` is a table (or tensor) with `n_classes` elements,
    such that `mapping[i]` is a table with 2 elements.
    * `mapping[i][1]` : index (1-based) of the cluster of class `i`
    * `mapping[i][2]` : index (1-based) of the index within its cluster of class `i`
*  `input_size` is the number of elements of the previous layer
*  `unk_index` is an index that is ignored at test time (not added to the
    loss). It can be disabled by setting it to 0 (not nil).
    It should only be used uring testing (since during training,
    it is not disabled in the backprop (TODO) )
]]
function HSM:__init(mapping, input_size, unk_index)
    parent.__init(self)
    if type(mapping) == 'table' then
        self.mapping = torch.LongTensor(mapping)
    else
        self.mapping = mapping:long()
    end
    self:check_mapping(self.mapping)
    self.n_classes = self.mapping:size(1)
    self.n_class_in_cluster = self:get_n_class_in_cluster(self.mapping)
    self.n_clusters = self.n_class_in_cluster:size(1)
    self.n_max_class_in_cluster = self.n_class_in_cluster:max()
    self.class_start_indices =
        torch.LongTensor(self.n_clusters):fill(0) -- 0 based !
    for i = 2, self.n_clusters do
        self.class_start_indices[i] =
            self.class_start_indices[i-1] + self.n_class_in_cluster[i-1]
    end
    self.unk_index      = unk_index or 0
    self.cluster_weight = torch.Tensor(self.n_clusters, input_size)
    self.cluster_bias   = torch.Tensor(self.n_clusters)
    self.cluster_score  = torch.Tensor(self.n_clusters)
    self.cluster_logsum = torch.Tensor(1)
    self.class_weight   = torch.Tensor(self.n_classes, input_size)
    self.class_bias     = torch.Tensor(self.n_classes)
    self.class_score    = torch.Tensor(self.n_max_class_in_cluster)
    self.class_logsum   = torch.Tensor(1)
    self.tmp_ones       = torch.Tensor(1):fill(1)
    self.gradInput      = torch.Tensor(input_size)
    self.cluster_grad_weight = torch.Tensor(self.n_clusters, input_size)
    self.cluster_grad_bias   = torch.Tensor(self.n_clusters)
    self.class_grad_weight   = torch.Tensor(self.n_classes, input_size)
    self.class_grad_bias     = torch.Tensor(self.n_classes)
    self.logSMCluster = nn.LogSoftMax()
    self.logLossCluster = nn.ClassNLLCriterion()
    self.logLossCluster.sizeAverage = false
    self.batch_size = 0
    self:reset()
end

function HSM:clone(...)
    return nn.Module.clone(self, ...)
end

function HSM:check_mapping(mapping)
    local n_classes = mapping:size(1)
    local clusters = {}
    for i = 1,n_classes do
        local cluster = mapping[i][1]
        local idx_in_cluster = mapping[i][2]
        clusters[cluster] = clusters[cluster] or {}
        table.insert(clusters[cluster], idx_in_cluster)
    end
    local cluster_max = 0
    for k, v in pairs(clusters) do
        cluster_max = math.max(cluster_max, k)
    end
    for i = 1, cluster_max do
        if clusters[i] == nil then
            error('HSM: bad mapping: not contiguous cluster indices (idx '
                      .. i .. ' is not skipped)')
        end
        table.sort(clusters[i])
        local last = 0
        for k, v in pairs(clusters[i]) do
            if k-1 ~= last then
                if k == last then
                    error('HSM: bad mapping: in cluster ' .. i
                              .. ', index ' .. k .. ' is used twice')
                else
                    error('HSM: bad mapping: in cluster ' .. i
                              .. ' indices are not contiguous (idx '
                              .. k .. ' is not skipped)')
                end
            end
            last = k
        end
    end
end

function HSM:get_n_class_in_cluster(mapping)
    local n_classes = mapping:size(1)
    local i_cluster_max = 0
    for i = 1, n_classes do
        i_cluster_max = math.max(i_cluster_max, mapping[i][1])
    end
    local cluster_counts = torch.LongTensor(i_cluster_max):zero()
    for i = 1, n_classes do
        cluster_counts[mapping[i][1] ] =
            cluster_counts[mapping[i][1] ] + 1
    end
    return cluster_counts
end

function HSM:parameters()
    return {self.cluster_weight, self.cluster_bias,
            self.class_weight, self.class_bias},
    {self.cluster_grad_weight, self.cluster_grad_bias,
     self.class_grad_weight, self.class_grad_bias}
end

function HSM:getParameters()
    return nn.Module.getParameters(self)
end

function HSM:reset(weight_stdv, bias_stdv)
    weight_stdv = weight_stdv or 0.1
    bias_stdv = bias_stdv or 0.1
    self.cluster_weight:normal():mul(weight_stdv)
    self.cluster_bias:normal():mul(bias_stdv)
    self.class_weight:normal():mul(weight_stdv)
    self.class_bias:normal():mul(bias_stdv)
end

function HSM:updateOutput(input, target)
    if self.cluster_weight:type() == 'torch.CudaTensor' then
        return self:updateOutputCUDA(input, target)
    else
        return self:updateOutputCPU(input, target)
    end
end

function HSM:updateOutputCPU(input, target)
    self.batch_size = input:size(1)
    if input:dim() == 1 then
        self.batch_size = 1
    else -- minibatch
        assert(input:dim() == 2)
    end
    if self.batch_size ~= self.tmp_ones:size(1) then
        self.tmp_ones:resize(self.batch_size):fill(1)
    end
    self.cluster_score:resize(self.batch_size, self.n_clusters)
    self.cluster_logsum:resize(self.batch_size)
    self.class_score:resize(self.batch_size, self.n_max_class_in_cluster)
    self.class_logsum:resize(self.batch_size)
    -- go through the cluster softmax
    -- linear layer
    if input:dim() == 1 then
       self.cluster_score:resize(self.cluster_bias:size(1))
       self.cluster_score:copy(self.cluster_bias)
       self.cluster_score:addmv(1, self.cluster_weight, input)
       loss = self.logLossCluster:forward(self.logSMCluster:forward(
                                             self.cluster_score),
                                          self.mapping[target[1]][1])
    else
      if self.n_clusters == 1 then
         self.cluster_score:zero():add(self.cluster_bias[1])
         self.cluster_score:select(2,1):addmv(1, input,
                                              self.cluster_weight:select(1,1))
         loss = self.logLossCluster:forward(
            self.logSMCluster:forward(self.cluster_score),
            torch.Tensor{self.mapping[target[1]][1]})
      else
         self.cluster_score:zero():addr(1, self.tmp_ones, self.cluster_bias)
         self.cluster_score:addmm(1, input, self.cluster_weight:t())
         loss = self.logLossCluster:forward(
            self.logSMCluster:forward(self.cluster_score),
            self.mapping:index(1, target):select(2,1))
      end
    end

    local n_valid
    self.output, n_valid = input.nn.HSM_updateOutputWithTarget(self, input,
                                                               target)
    n_valid = self.batch_size --TODO
    assert(self.unk_index == 0)
    self.output = self.output + loss
    return self.output, n_valid
end

function HSM:updateOutputCUDA(input, target)
    if (input:type() ~= 'torch.CudaTensor') or
    (target:type() ~= 'torch.CudaTensor') then
        error('CudaTensor expected')
    end
    self.batch_size = input:size(1)
    assert(input:dim() ~= 1)
    if self.batch_size ~= self.tmp_ones:size(1) then
        self.tmp_ones:resize(self.batch_size):fill(1)
    end
    self.cluster_score:resize(self.batch_size, self.n_clusters)
    self.cluster_logsum:resize(self.batch_size)
    self.class_score:resize(self.batch_size, self.n_max_class_in_cluster)
    self.class_logsum:resize(self.batch_size)
    -- go through the cluster softmax
    -- linear layer
    self.cluster_score:zero():addr(1, self.tmp_ones, self.cluster_bias)
    self.cluster_score:addmm(1, input, self.cluster_weight:t())

    if (type(self.output) == 'number') or
        (self.output:type() ~= 'torch.CudaTensor') then
            self.output = torch.CudaTensor(1)
    end
    self.output:zero()
    input.nn.HSM_updateOutputWithTarget(self, input, target)

    assert(self.unk_index == 0) --TODO
    return self.output, self.batch_size
end

-- Note: call this function at most once after each call `updateOutput`,
-- or the output will be wrong (it uses `class_score` and `cluster_score`
-- as temporary buffers)
function HSM:updateGradInput(input, target)
    if input:type() == 'torch.CudaTensor' then
        return self:updateGradInputCUDA(input, target)
    else
        return self:updateGradInputCPU(input, target)
    end
end

function HSM:updateGradInputCPU(input, target)
    if self.unk_index ~= 0 then
        error("HSM: bprop with unk is not implemented")
    end
    self.gradInput:resizeAs(input)
    -- BPROP through the cluster prediction
    local clusterID = self.mapping:index(1, target):select(2,1)
    if input:dim() == 1 then
       clusterID = clusterID[1]
    end
    self.logLossCluster:updateGradInput(self.logSMCluster.output,
                                        clusterID)
    self.logSMCluster:updateGradInput(self.cluster_score,
                                      self.logLossCluster.gradInput)
    if input:dim() == 1 then
       self.gradInput:addmv(
          0, 1, self.cluster_weight:t(), self.logSMCluster.gradInput)
    else
       self.gradInput:addmm(
          0, 1, self.logSMCluster.gradInput, self.cluster_weight)
    end
    input.nn.HSM_updateGradInput(self, target)
    return self.gradInput
end

function HSM:updateGradInputCUDA(input, target)
    self.gradInput:resizeAs(input)
    assert(input:dim() == 2)
    input.nn.HSM_updateGradInput(self, target)
    self.gradInput:addmm(1, 1, self.cluster_score, self.cluster_weight)
    return self.gradInput
end

-- If `direct_update` is set, the parameters are directly updated (not the
-- gradients). It means that the gradient tensors (like `cluster_grad_weight`)
-- are not used. scale must be set to the negative learning rate
-- (`-learning_rate`). `direct_update` mode is much faster.
-- Before calling this function you have to call `HSM:updateGradInput` first.
function HSM:accGradParameters(input, target, scale, direct_update)
    scale = scale or 1
    if self.unk_index ~= 0 then
        error("HSM: bprop with unk is not implemented")
    end

    local cluster_gradInput = self.logSMCluster.gradInput
    if input:type() == 'torch.CudaTensor' then
        cluster_gradInput = self.cluster_score
    end

    if direct_update then
       if input:dim() == 1 then
          self.cluster_bias:add(scale, cluster_gradInput)
          self.cluster_weight:addr(scale, cluster_gradInput, input)
       else
          if self.n_clusters == 1 then
             self.cluster_weight:select(1,1):addmv(
                scale, input:t(), cluster_gradInput:select(2,1))
             self.cluster_bias:addmv(
                scale, cluster_gradInput:t(),
                self.tmp_ones)
          else
             self.cluster_weight:addmm(scale, cluster_gradInput:t(), input)
             self.cluster_bias:addmv(scale, cluster_gradInput:t(),
                                     self.tmp_ones)
          end
       end
       input.nn.HSM_accGradParameters_directUpdate(self, input, target, scale)
    else
       if input:dim() == 1 then
          self.cluster_grad_bias:add(scale, cluster_gradInput)
          self.cluster_grad_weight:addr(scale, cluster_gradInput, input)
       else
          if self.n_clusters == 1 then
             self.cluster_grad_weight:select(1,1):addmv(
                scale, input:t(), cluster_gradInput:select(2,1))
             self.cluster_grad_bias:addmv(
                scale, cluster_gradInput:t(), self.tmp_ones)
          else
             self.cluster_grad_weight:addmm(scale, cluster_gradInput:t(), input)
             self.cluster_grad_bias:addmv(scale, cluster_gradInput:t(),
                                          self.tmp_ones)
          end
       end
       input.nn.HSM_accGradParameters(self, input, target, scale)
    end
end

function HSM:backward(input, target, scale)
    self:updateGradInput(input, target)
    self:accGradParameters(input, target, scale)
    return self.gradInput
end

function HSM:updateParameters(learning_rate)
    self.cluster_weight:add(-learning_rate, self.cluster_grad_weight)
    self.cluster_bias  :add(-learning_rate, self.cluster_grad_bias  )
    self.class_weight  :add(-learning_rate, self.class_grad_weight  )
    self.class_bias    :add(-learning_rate, self.class_grad_bias    )
end

function HSM:zeroGradParameters()
    self.cluster_grad_weight:zero()
    self.cluster_grad_bias:zero()
    self.class_grad_weight:zero()
    self.class_grad_bias:zero()
end

function HSM:zeroGradParametersClass(input, target)
   input.nn.HSM_zeroGradParametersClass(self, target)
end

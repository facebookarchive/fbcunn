-- Copyright 2004-present Facebook. All Rights Reserved.
-- Author: Michael Mathieu <myrhev@fb.com>

require 'fbcunn'
require 'fbnn'

local function test_finite_diff_gradInput(model, input, target)
    local eps = 1e-3
    local output = model:updateOutput(input, target)
    local gradInput = model:updateGradInput(input, target):clone()

    local gradInput2 = torch.Tensor(input:size())
    if input:dim() == 1 then
        for i = 1,input:size(1) do
            input[i] = input[i] + eps
            local outputP = model:updateOutput(input, target)
            input[i] = input[i] - 2*eps
            local outputM = model:updateOutput(input, target)
            input[i] = input[i] + eps
            gradInput2[i] = (outputP - outputM) / (2*eps)
        end
    else
        assert(input:dim() == 2)
        for i = 1,input:size(1) do
            for j = 1,input:size(2) do
                input[i][j] = input[i][j] + eps
                local outputP = model:updateOutput(input, target)
                input[i][j] = input[i][j] - 2*eps
                local outputM = model:updateOutput(input, target)
                input[i][j] = input[i][j] + eps
                gradInput2[i][j] = (outputP - outputM) / (2*eps)
            end
        end
    end
    return (gradInput - gradInput2):abs():max()
end

local function test_finite_diff_accGrads(model, input, target, scale)
    local eps = 1e-3
    scale = scale or 1

    local w, dw = model:getParameters()

    dw:zero()
    local output = model:updateOutput(input, target)
    local gradInput = model:updateGradInput(input, target):clone()
    model:accGradParameters(input, target, scale)
    local gradParams = dw:clone()

    local gradParams2 = torch.Tensor(w:size(1))
    for i = 1,w:size(1) do
        w[i] = w[i] + eps
        local outputP = model:updateOutput(input, target)
        w[i] = w[i] - 2*eps
        local outputM = model:updateOutput(input, target)
        w[i] = w[i] + eps
        gradParams2[i] = scale * (outputP - outputM) / (2*eps)
    end

    return (gradParams - gradParams2):abs():max()
end

for i = 1,100 do
   print("Iteration " .. i)
    local n_clusters = torch.random(10)
    local n_class = torch.random(50) + n_clusters - 1
    local mapping = {}
    local n_class_in_cluster = {}
    for i = 1, n_class do
        local cluster = torch.random(n_clusters)
        n_class_in_cluster[cluster] = n_class_in_cluster[cluster] or 0
        n_class_in_cluster[cluster] = n_class_in_cluster[cluster] + 1
        mapping[i] = {cluster, n_class_in_cluster[cluster]}
    end
    for i = 1,n_clusters do
        if n_class_in_cluster[i] == nil then
            n_class_in_cluster[i] = 1
            mapping[1+#mapping] = {i, 1}
            n_class = n_class + 1
        end
    end
    local input_size = torch.random(100) + 1
    local model = nn.HSM(mapping, input_size)

    local input = torch.randn(input_size)
    local target = torch.LongTensor(1)
    target[1] = torch.random(n_class)
    local err = test_finite_diff_gradInput(model, input, target)
    assert(err < 1e-2)
    err = test_finite_diff_accGrads(model, input, target)
    assert(err < 1e-2)
    local scale = torch.rand(1)[1]
    err = test_finite_diff_accGrads(model, input, target, scale)
    assert(err < 1e-2)

    local batch_size = torch.random(10)
    input = torch.randn(batch_size, input_size)
    target = torch.LongTensor(batch_size)
    for i = 1, batch_size do
        target[i] = torch.random(n_class)
    end
    err = test_finite_diff_gradInput(model, input, target);
    assert(err < 1e-2)
    err = test_finite_diff_accGrads(model, input, target)
    assert(err < 1e-2)
    err = test_finite_diff_accGrads(model, input, target, scale)
    assert(err < 1e-2)

    -- test directUpdate
    local w, dw = model:getParameters()
    dw:normal()
    local initdw = dw:clone()
    model:updateOutput(input, target)
    model:updateGradInput(input, target)
    model:accGradParameters(input, target, scale, false)
    local w1 = w:clone():add(dw)
    model:updateOutput(input, target)
    model:updateGradInput(input, target)
    model:accGradParameters(input, target, scale, true)
    w:add(initdw)
    err = w:add(-1, w1):abs():max()
    assert(err < 1e-5)
end

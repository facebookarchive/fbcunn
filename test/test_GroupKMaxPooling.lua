-- Copyright 2004-present Facebook. All Rights Reserved.

require('fb.luaunit')

require('math')

require('fbtorch')

require('nn')

require('fbcunn')
require('fbnn')

function run_GroupKMaxPooling_updateOutput(n, d, k)
    -- n = number of words
    -- d = dimension of embeddings
    -- k = k-max pooling
    local input = torch.randn(n, d)
    local kmax = nn.GroupKMaxPooling(k)

    local output = kmax:updateOutput(input)

    assert(output == kmax.output)
    assert(output:size(1) == k)
    assert(output:size(2) == input:size(2))

    local norms = torch.norm(input, 2, 2)
    local _, kmax_indices = torch.sort(norms, 1)
    kmax_indices = kmax_indices[{{-k,-1}}]
    kmax_indices = torch.sort(kmax_indices, 1)

    local kmax_result = torch.Tensor(k, input:size(2))
    for i = 1, kmax_indices:size(1) do
        kmax_result:select(1, i):copy(input:select(1, kmax_indices[i][1]))
    end

    assert(torch.sum(torch.eq(kmax_result, output)) == torch.numel(output))
end

function test_GroupKMaxPooling_updateOutput()
    run_GroupKMaxPooling_updateOutput(10, 11, 4)
end

function run_GroupKMaxPooling_updateOutput_batch(b, n, d, k)
    -- b = batch size
    -- n = number of words
    -- d = dimension of embeddings
    -- k = k-max pooling
    local input = torch.randn(b, n, d)
    local kmax = nn.GroupKMaxPooling(k)

    local output = kmax:updateOutput(input)

    assert(output == kmax.output)
    assert(output:size(1) == b)
    assert(output:size(2) == k)
    assert(output:size(3) == input:size(3))

    local norms = torch.norm(input, 2, 3):squeeze()
    local _, kmax_indices = torch.sort(norms, 2)
    kmax_indices = kmax_indices:sub(1, -1, -k, -1)
    kmax_indices = torch.sort(kmax_indices, 2)

    local kmax_result = torch.Tensor(input:size(1), k, input:size(3))
    kmax_result:fill(0.0)

    for i = 1, kmax_indices:size(1) do
        for j = 1, kmax_indices:size(2) do
            kmax_result:select(1, i):select(1, j):copy(
                input:select(1, i):select(1, kmax_indices[i][j]))
        end
    end

    assert(torch.sum(torch.eq(kmax_result, output)) == torch.numel(output))
end

function test_GroupKMaxPooling_updateOutput_batch()
    run_GroupKMaxPooling_updateOutput_batch(15, 10, 11, 4)
end

function run_GroupKMaxPooling_updateGradInput(n, d, k)
    -- n = number of words
    -- d = dimension of embeddings
    -- k = k-max pooling
    local input = torch.randn(n, d)

    local kmax = nn.GroupKMaxPooling(k)

    local output = kmax:updateOutput(input)

    local delta = torch.randn(output:size())

    local gradInput = kmax:updateGradInput(input, delta)

    assert(gradInput == kmax.gradInput)

    assert(gradInput:sum() == delta:sum())
end

function test_GroupKMaxPooling_updateGradInput()
    run_GroupKMaxPooling_updateOutput(10, 11, 4)
end


function run_GroupKMaxPooling_updateGradInput_batch(b, n, d, k)
    -- n = number of words
    -- d = dimension of embeddings
    -- k = k-max pooling
    local input = torch.randn(b, n, d)

    local kmax = nn.GroupKMaxPooling(k)

    local output = kmax:updateOutput(input)

    local delta = torch.randn(output:size())

    local gradInput = kmax:updateGradInput(input, delta)

    assert(gradInput == kmax.gradInput)

    assert(gradInput:sum() == delta:sum())
end

function test_GroupKMaxPooling_updateGradInput_batch()
    run_GroupKMaxPooling_updateOutput(12, 10, 11, 4)
end

LuaUnit:main()

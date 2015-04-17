-- Copyright 2004-present Facebook. All Rights Reserved.

require('fb.luaunit')

require('math')

require('fbtorch')

require('nn')

require('fbcunn')
require('fbnn')

local n_test_repeats = 1

function run_KMaxPooling_updateOutput(n, d, k, infer_length)
    -- n = number of words
    -- d = dimension of embeddings
    -- k = k-max pooling
    -- infer_length = if true, have the module infer the length from the batch

    local input = torch.randn(n, d)
    local input_info = nil
    local output_info = nil
    local output = nil
    local output_pack = nil

    if not infer_length then
        local input_length = torch.LongTensor(1)
        input_length[1] = n
        input_info = { length = input_length }
    end

    local kmax = nn.KMaxPooling(k)

    local output_pack = kmax:updateOutput(input, input_info)

    if input_info ~= nil then
        output = output_pack.output
        output_info = output_pack.info
    else
        output = output_pack
    end

    assert(output == kmax.output)

    if input_info ~= nil then
        assert(input_info.length[1] == n)
        assert(output_info.length[1] == math.min(n, k))
    end

    assert(kmax.input_length[1] == n)
    assert(kmax.output_length[1] == math.min(n, k))

    assert(kmax.output:size(1) == k)
    assert(kmax.output:size(2) == input:size(2))

    local kth_max = torch.min(kmax.output, 1)
    local kth_max = kth_max:expand(input:size(1), input:size(2));

    local count_at_least_kth_max = torch.le(kth_max, input):sum(1)

    assert(torch.eq(count_at_least_kth_max, math.min(n, k)):sum() == d)
end

function test_KMaxPooling_updateOutput_longpooling_inferlength()
    for i = 1, n_test_repeats do
        run_KMaxPooling_updateOutput(10, 10, 30, true)
    end
end

function test_KMaxPooling_updateOutput_shortpooling_inferlength()
    for i = 1, n_test_repeats do
        run_KMaxPooling_updateOutput(100, 10, 30, true)
    end
end

function test_KMaxPooling_updateOutput_longpooling()
    for i = 1, n_test_repeats do
        run_KMaxPooling_updateOutput(10, 10, 30, false)
    end
end

function test_KMaxPooling_updateOutput_shortpooling()
    for i = 1, n_test_repeats do
        run_KMaxPooling_updateOutput(100, 10, 30, false)
    end
end


function run_KMaxPooling_updateGradInput(n, d, k, infer_length)
    -- n = number of words
    -- d = dimension of embeddings
    -- k = k-max pooling
    -- infer_length = if true, have the module infer the length from the batch

    local input = torch.randn(n, d)
    local input_info = nil
    local output_info = nil
    local output = nil
    local gradInput_pack = nil
    local gradInput = nil
    local gradInput_info = nil

    if not infer_length then
        local input_length = torch.LongTensor(1)
        input_length[1] = n
        input_info = { length = input_length }
    end

    local kmax = nn.KMaxPooling(k)

    local output_pack = kmax:updateOutput(input, length, input_info)

    if input_info ~= nil then
        output = output_pack.output
        output_info = output_pack.info
    else
        output = output_pack
    end

    local delta = torch.randn(kmax.output:size())

    local gradInput = kmax:updateGradInput(input, delta)

    assert(gradInput == kmax.gradInput)

    if input_info ~= nil then
        assert(input_info.length[1] == n)
    end

    assert(kmax.gradInput:size(1) == input:size(1))
    assert(kmax.gradInput:size(2) == input:size(2))

    local grad_input_sum = torch.sum(kmax.gradInput)
    local delta_sum = torch.sum(delta[{{1,math.min(n,k)}}])

    assert(math.abs(grad_input_sum - delta_sum) < 1e-6)
end

function test_KMaxPooling_updateGradInput_longpooling_inferlength()
    for i = 1, n_test_repeats do
        run_KMaxPooling_updateGradInput(100, 10, 30, true)
    end
end

function test_KMaxPooling_updateGradInput_shortpooling_inferlength()
    for i = 1, n_test_repeats do
        run_KMaxPooling_updateGradInput(10, 10, 30, true)
    end
end

function test_KMaxPooling_updateGradInput_longpooling()
    for i = 1, n_test_repeats do
        run_KMaxPooling_updateGradInput(100, 10, 30, false)
    end
end

function test_KMaxPooling_updateGradInput_shortpooling()
    for i = 1, n_test_repeats do
        run_KMaxPooling_updateGradInput(10, 10, 30, false)
    end
end

function test_KMaxPooling_inside_Sequential()
    local input = torch.randn(10, 5)
    local seq = nn.Sequential()

    local kmax = nn.KMaxPooling(5)

    seq:add(nn.KMaxPooling(5))

    local kmax_output = kmax:updateOutput(input)
    local seq_output = seq:updateOutput(input)
    local output_matches = torch.eq(kmax_output, seq_output)
    assert (output_matches:sum() == output_matches:numel())

    local delta = torch.randn(kmax.output:size())

    local kmax_gradInput = kmax:updateGradInput(input, delta)
    local seq_gradInput = kmax:updateGradInput(input, delta)
    local gradInput_matches = torch.eq(kmax_output, seq_output)
    assert (gradInput_matches:sum() == gradInput_matches:numel())


end

LuaUnit:main()

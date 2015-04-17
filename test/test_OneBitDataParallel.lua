require('fb.luaunit')
require('fbtorch')
require('fbcunn')
require('fbnn')
local TU = require('test.test_Util')
local fboptim = require('fboptim')

local function dp()
    return nn.OneBitDataParallel(
        1,
        {momentum_rate=1.0, adagrad_learning_rate=1.0, min_elements=20}
    )
end


function testDataParallelRunsForwardPass()
    local sim = TU.Sim {
        num_hidden = 2,
        output_width = 1,
        hidden_width = 512,
        input_width = 32,
        num_columns = 4,
    }

    local model, columns = sim:build_data_parallel(dp())
    local inputs, _ = sim:gen_wide_example()
    local outputs = model:forward(inputs)

    for column_id = 1, sim.opts.num_columns do
        local column_input = sim:get_narrowed_input(inputs, column_id):double()
        print(column_input:size())
        local gpu_output = outputs[{ {column_id} }]
        local cpu_output = columns[column_id]:forward(column_input)

        local norm_delta = TU.tensor_norm_difference(gpu_output, cpu_output)

        print(column_input:size(), gpu_output:size(), cpu_output:size())
        print(gpu_output:norm(), cpu_output:norm())
        assertTrue(norm_delta < 1E-5)
    end
end

function testDataParallelOnForwardPassIsEquivalentToSeparateColumns()
    local sim = TU.Sim {
        num_hidden = 2,
        output_width = 1,
        hidden_width = 512,
        input_width = 32,
        num_columns = 4,
    }

    local model, columns = sim:build_data_parallel(dp())
    local inputs, _ = sim:gen_wide_example()
    local outputs = model:forward(inputs)

    for column_id = 1, sim.opts.num_columns do
        local column_input = sim:get_narrowed_input(inputs, column_id):double()
        print(column_input:size())
        local gpu_output = outputs[{ {column_id} }]
        local cpu_output = columns[column_id]:forward(column_input)

        local norm_delta =
            TU.tensor_norm_difference(gpu_output, cpu_output)

        print(column_input:size(), gpu_output:size(), cpu_output:size())
        print(gpu_output:norm(), cpu_output:norm())
        assertTrue(norm_delta < 1E-5)
    end
end

function testDataParallelOnOptimLearns()
    local sim = TU.Sim {
        num_hidden = 1,
        output_width = 1,
        hidden_width = 500,
        input_width = 5,
        num_columns = 4,
        num_opt_rounds = 2,
    }

    local optim_state = {
        learningRate = 1e-1,
        weightDecay = 1e-4,
        momentum = 0.9,
        learningRateDecay = 1e-7
    }

    local model, _columns = sim:build_data_parallel(dp())
    local opt = nn.Optim(model, optim_state)
    local criterion = nn.MSECriterion():cuda()

    for round = 1,sim.opts.num_opt_rounds do
        local inputs, targets = sim:gen_wide_example()
        local _outputs = model:forward(inputs)
        opt:optimize(fboptim.sgd, inputs, targets, criterion)
        local out = model:forward(inputs)
        print(out)
        local err = criterion:forward(out, targets)
        print(round, err)
    end
end

LuaUnit:main()

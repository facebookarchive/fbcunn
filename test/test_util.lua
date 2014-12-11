require('fb.luaunit')
require('fbtorch')
require('cunn')
require('optim')

local M = {}

local pl = require('pl.import_into')()

function M.tensor_norm_difference(l, r)
    return math.abs(l:norm() - r:norm())
end

function M.assertTensorEquals(a, b)
    assertEquals(0, (a:clone():add(b:clone():mul(-1))):abs():sum())
end

function M.assertTensorAlmostEquals(a, b, eps)
    assertTrue((a:clone():add(b:clone():mul(-1))):norm() < (eps or 1E-10))
end

local Sim = pl.class()
M.Sim = Sim

function Sim:_init(opts)
    self.opts = opts
end

function Sim:build_column()
    local seq = nn.Sequential()
    local pred = self.opts.input_width
    for i = 1,self.opts.num_hidden do
        seq:add(nn.Linear(pred, self.opts.hidden_width))
        seq:add(nn.Tanh())
        pred = self.opts.hidden_width
    end
    seq:add(nn.Linear(self.opts.hidden_width, self.opts.output_width))
    seq:add(nn.Tanh())
    return seq
end

function Sim:build_data_parallel(dp)
    local num_gpus = cutorch.getDeviceCount()
    local columns = {}

    for column_id = 1,self.opts.num_columns do
        local gpu_id = column_id % num_gpus
        if gpu_id == 0 then gpu_id = num_gpus end
        print(gpu_id)
        cutorch.withDevice(
            gpu_id,
            function()
                local column = self:build_column()
                table.insert(columns, column:clone())
                dp:add(column:clone(), gpu_id)
            end
        )
    end
    return dp:cuda(), columns
end

function Sim:target_function(x)
    -- admittedly tough for us to learn, but hey.
    local retval = torch.Tensor(self.opts.output_width)
    local sum = x:sum()
    retval[1] = math.sin(sum)
    return retval
end

function Sim:gen_wide_input()
    return torch.randn(self.opts.input_width * self.opts.num_columns)
end

function Sim:get_narrowed_input_range(i)
    assert(type(i) == 'number')
    local range_start = 1 + ((i - 1) * self.opts.input_width)
    local range_end = range_start + (self.opts.input_width) - 1
    return range_start, range_end
end

function Sim:get_narrowed_input(input, i)
    assert(torch.typename(input))
    assert(type(i) == 'number')
    return input[{ {self:get_narrowed_input_range(i)} }]
end

function Sim:gen_wide_example()
    local samp = self:gen_wide_input()
    local retval = torch.Tensor(self.opts.output_width * self.opts.num_columns)
    for i = 1,self.opts.num_columns do
        retval[i] = self:target_function(self:get_narrowed_input(samp, i))
    end
    return samp:cuda(), retval:cuda()
end

return M

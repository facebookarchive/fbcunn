-- Copyright 2004-present Facebook. All Rights Reserved.

require('fbtorch')
require('fb.luaunit')
require('fbcunn')

require('nn')

local function all(tensor)
    return torch.sum(torch.ne(tensor, 0)) == tensor:numel()
end

local function almost_equal(t1, t2, tol)
    return torch.lt(torch.abs(t1 - t2), tol)
end

-- w = weighted
-- u = unweighted
-- e.g.
-- wlut = weighted lookup table
-- ulut = unweighted lookup table

function test_WeightedLookupTable_forward()
    local embedding_dim = 4
    local table_size = 30
    local input_length = 9
    local tol = 1e-8

    local wlut = nn.WeightedLookupTable(table_size, embedding_dim):cuda()
    local ulut = nn.LookupTable(table_size, embedding_dim):cuda()
    ulut.weight:copy(wlut.weight)
    assert(all(torch.eq(wlut.weight, ulut.weight)))

    local uinput = torch.rand(input_length):mul(table_size):ceil()
    local weights = torch.rand(input_length, 1)
    local winput = torch.cat(uinput, weights, 2)

    local woutput = wlut:forward(winput:cuda())
    local uoutput = ulut:forward(uinput:cuda())
    weights = weights:cuda()
    local expected_woutput = torch.cmul(uoutput, weights:expandAs(uoutput))

    assert(all(almost_equal(woutput:float(), expected_woutput:float(), tol)))
end

function test_WeightedLookupTable_accGradParameters()
    local embedding_dim = 4
    local table_size = 30
    local input_length = 9
    local tol = 1e-5

    local wlut = nn.WeightedLookupTable(table_size, embedding_dim):cuda()
    local ulut = nn.LookupTable(table_size, embedding_dim):cuda()
    ulut.weight:copy(wlut.weight)
    assert(all(torch.eq(wlut.weight, ulut.weight)))

    local uinput = torch.rand(input_length):mul(table_size):ceil()
    local weights = torch.range(1, input_length):reshape(input_length, 1)
    local winput = torch.cat(uinput, weights, 2)

    winput = winput:cuda()
    uinput = uinput:cuda()
    local woutput = wlut:forward(winput)
    local uoutput = ulut:forward(uinput)

    local wgradOutput = torch.randn(woutput:size())
    local ugradOutput = torch.cmul(wgradOutput, weights:expandAs(wgradOutput))

    wlut:accGradParameters(winput, wgradOutput:cuda(), 1)
    ulut:accGradParameters(uinput, ugradOutput:cuda(), 1)

    assert(all(almost_equal(wlut.gradWeight:float(), ulut.gradWeight:float(), tol)))
end


LuaUnit:main()

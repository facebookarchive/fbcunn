require 'nn'
require 'cunn'
require 'fbtorch'
require 'fbcunn'

local mytester = torch.Tester()
local LinearNBTest = {}
local jac = nn.Jacobian

local precision = 1e-5

function testAll(targettype)
    targettype = targettype or 'torch.DoubleTensor'
    local ini = math.random(3,5)
    local inj_vals = {math.random(3,5), 1}  -- Also test the inj = 1 spatial case
    local input = torch.Tensor(ini):zero():type(targettype)

    for ind, inj in pairs(inj_vals) do
        local module = nn.LinearNB(ini, inj)
        if targettype == 'torch.CudaTensor' then
            module = module:cuda()
        end

        -- 1D
        local err = jac.testJacobian(module, input)
        mytester:assertlt(err, precision, 'error on state ')

        local err = jac.testJacobianParameters(module, input, module.weight,
                                               module.gradWeight)
        mytester:assertlt(err, precision, 'error on weight ')

        local err = jac.testJacobianUpdateParameters(module, input, module.weight)
        mytester:assertlt(err, precision, 'error on weight [direct update] ')

        for t,err in pairs(jac.testAllUpdate(module, input,
                                             'weight', 'gradWeight')) do
            mytester:assertlt(err, precision, string.format(
                                  'error on weight [%s]', t))
        end

        -- 2D
        local nframe = math.random(50,70)
        local input = torch.Tensor(nframe, ini):zero():type(targettype)

        local err = jac.testJacobian(module,input)
        mytester:assertlt(err, precision, 'error on state ')

        local err = jac.testJacobianParameters(module, input, module.weight,
                                               module.gradWeight)
        mytester:assertlt(err,precision, 'error on weight ')

        local err = jac.testJacobianUpdateParameters(module, input, module.weight)
        mytester:assertlt(err,precision, 'error on weight [direct update] ')

        for t,err in pairs(jac.testAllUpdate(module, input,
                                             'weight', 'gradWeight')) do
            mytester:assertlt(err, precision, string.format(
                                  'error on weight [%s]', t))
        end

        -- IO
        local ferr,berr = jac.testIO(module, input)
        mytester:asserteq(ferr, 0, torch.typename(module)
                              .. ' - i/o forward err ')
        mytester:asserteq(berr, 0, torch.typename(module)
                              .. ' - i/o backward err ')
    end
end

function LinearNBTest.testDouble()
    testAll()
end


mytester:add(LinearNBTest)
mytester:run()

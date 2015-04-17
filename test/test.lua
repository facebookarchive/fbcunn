require 'cunn'
require 'fbcunn'
require 'nn'
require 'fbnn'

local mytester = torch.Tester()
local fbcunntest = {}
local precision_forward = 1e-4
local times = {}

function fbcunntest.TemporalMaxPooling()
   -- non-batch mode: compare against nn.TemporalMaxPooling
   local from = math.random(50, 70)
   local ki = math.random(5, 10)
   local si = math.random(1, 2)
   local outi = math.random(50, 90)
   local ini = (outi - 1) * si + ki
   local module = nn.TemporalMaxPooling(ki, si)
   local cudaModule = nn.TemporalMaxPooling(ki, si):cuda()

   local tm = {}
   local title =
      string.format('TemporalMaxPooling.forward %dx%d ',
                    ini, from)
   times[title] = tm

   local input = torch.randn(ini, from)
   local inputCuda = input:cuda()

   local a = torch.Timer()
   local output = module:forward(input)
   tm.cpu = a:time().real

   a:reset()
   local outputCuda = cudaModule:forward(inputCuda)
   cutorch.synchronize()
   tm.gpu = a:time().real

   mytester:assertTensorEq(output:float(),
                           outputCuda:float(), 1e-4, 'error on forward')

   tm = {}
   title =
      string.format('TemporalMaxPooling.backward %dx%d ',
                    ini, from)
   times[title] = tm

   local gradOutput = torch.randn(outi, from)
   local gradOutputCuda = gradOutput:cuda()

   a:reset()
   local gradInput = module:backward(input, gradOutput)
   tm.cpu = a:time().real

   a:reset()
   local gradInputCuda = cudaModule:backward(inputCuda, gradOutputCuda)
   cutorch.synchronize()
   tm.gpu = a:time().real

   mytester:assertTensorEq(gradInput:float(),
                           gradInputCuda:float(), 1e-4, 'error on backward')
end

function fbcunntest.TemporalMaxPoolingBatch()
   -- batch mode: compare against nn.TemporalMaxPooling
   local from = math.random(50, 70)
   local ki = math.random(5, 10)
   local si = math.random(1, 2)
   local outi = math.random(50, 90)
   local batchSize = math.random(2, 5)
   local ini = (outi - 1) * si + ki
   local module = nn.TemporalMaxPooling(ki, si)
   local cudaModule = nn.TemporalMaxPooling(ki, si):cuda()

   local tm = {}
   local title =
      string.format('TemporalMaxPooling.forward (batch) %dx%dx%d ',
                    batchSize, ini, from)
   times[title] = tm

   local input = torch.randn(batchSize, ini, from)
   local inputCuda = input:cuda()

   local a = torch.Timer()
   local output = module:forward(input)
   tm.cpu = a:time().real

   a:reset()
   local outputCuda = cudaModule:forward(inputCuda)
   cutorch.synchronize()
   tm.gpu = a:time().real

   mytester:assertTensorEq(output:float(),
                           outputCuda:float(),
                           1e-4, 'error on forward batch')

   local gradOutput = torch.randn(batchSize, outi, from)
   local gradOutputCuda = gradOutput:cuda()

   tm = {}
   title =
      string.format('TemporalMaxPooling.backward (batch) %dx%dx%d ',
                    batchSize, ini, from)
   times[title] = tm

   a:reset()
   local gradInput = module:backward(input, gradOutput)
   tm.cpu = a:time().real

   a:reset()
   local gradInputCuda = cudaModule:backward(inputCuda, gradOutputCuda)
   cutorch.synchronize()
   tm.gpu = a:time().real

   mytester:assertTensorEq(gradInput:float(),
                           gradInputCuda:float(),
                           1e-4, 'error on backward batch')
end

function fbcunntest.Optim()
    require 'cunn'
    local fboptim = require 'fboptim'

    -- Goofy artificial problem that tests that the optim module
    -- works.
    local numInputs = 20
    local numHidden = 100
    local numHidden2 = 200
    local numOut = 2

    local model = nn.Sequential()
    model:add(nn.Linear(numInputs, numHidden))
    model:add(nn.Threshold())
    model:add(nn.Linear(numHidden, numHidden2))
    model:add(nn.Threshold())
    model:add(nn.Linear(numHidden2, numOut))
    model = model:cuda()

    local optimState = {
        learningRate = 0.001,
        weightDecay = 0.0001,
        momentum = 0.9,
        learningRateDecay = 1e-6
    }

    local PO = nn.Optim(model, optimState)

    local function getTrainingSample()
        local input = torch.randn(numInputs)
        local output = torch.randn(2)
        output[1] = input:sum()
        output[2] = input[1] - input[numInputs]
        return input, output
    end

    local function getTrainingBatch(size)
        local ins = torch.Tensor(size, numInputs)
        local outs = torch.Tensor(size, numOut)
        for i = 1,size do
            ins[i], outs[i] = getTrainingSample()
        end
        return ins:cuda(), outs:cuda()
    end

    local crit = nn.MSECriterion():cuda()
    local err = 0.0
    for i=1,1000 do
        local inp, outp = getTrainingBatch(1024)
        err = PO:optimize(fboptim.sgd, inp, outp, crit)
    end
    assert(err < 1.0)
end

function fbcunntest.Module_getParametersByDevice()
    if cutorch.getDeviceCount() == 0 then
        return
    end

    local seq = nn.Sequential()
    local mp = nn.ModelParallel(2)
    mp:add(nn.Linear(100, 5)) -- GPU 1
    mp:add(nn.Linear(100, 5)) -- GPU 2
    mp = mp:cuda()
    seq:add(mp)
    seq:add(nn.Linear(5, 5)) -- CPU

    local params_by_dev, grads_by_dev = seq:getParametersByDevice()

    local function lin_param_size(ins, outs)
        return ins * outs + outs
    end
    local function assert_on_dev_with_size(parms, dev, ins, outs)
        assert(parms)
        assert(#parms:size() == 1)
        assert(parms:size()[1] == lin_param_size(ins, outs))
        assert(dev == 0 or parms:getDevice() == dev)
    end
    assert_on_dev_with_size(params_by_dev[0], 0, 5, 5)
    assert_on_dev_with_size(grads_by_dev[0], 0, 5, 5)

    if cutorch.getDeviceCount() >= 2 then
        assert_on_dev_with_size(params_by_dev[1], 1, 100, 5)
        assert_on_dev_with_size(grads_by_dev[1],  1, 100, 5)
        assert_on_dev_with_size(params_by_dev[2], 2, 100, 5)
        assert_on_dev_with_size(grads_by_dev[2],  2, 100, 5)
    else
        assert_on_dev_with_size(params_by_dev[1], 1, 200, 10)
        assert_on_dev_with_size(grads_by_dev[1],  1, 200, 10)
    end
end

mytester:add(fbcunntest)

function nn.testfbcunn(tests)
    local oldtype = torch.getdefaulttensortype()
    torch.setdefaulttensortype('torch.FloatTensor')
    math.randomseed(os.time())
    mytester:run(tests)
    torch.setdefaulttensortype(oldtype)
end

require 'fb.luaunit'
require 'torch'
require 'cunn'
require 'sys'
require 'fbcunn'

function testEndToEnd()
    torch.setdefaulttensortype('torch.FloatTensor')

    local mptest = {}
    local utils = {}

    local times = {}
    local nGPUs = 4
    local precision = 1e-4

    local mytester = torch.Tester()

    function utils.forEachGPU(closure)
       local curDevice = cutorch.getDevice()
       for i=1,nGPUs do
          cutorch.withDevice(i, closure)
       end
    end

    function utils.initializeGPUs()
      cutorch.manualSeedAll(os.time())
    end

    function utils.timedRun(func, ...)
       local t = torch.Timer()
       func(...)
       utils.forEachGPU(function()
          cutorch.synchronize()
       end)
       local time = t:time().real
       return time
    end

    local function testSpatialConvolution(testModule, name)
       local scale = 1
       local learning_rate = 0.1

       local to = {}
       for i=1,nGPUs do
          table.insert(to, 16)
       end
       local from = 3
       local ki = 11
       local kj = 11
       local si = 1
       local sj = 1
       local ini = 128
       local inj = 128
       local batch = 32

       local splitDim = 1
       local concatDim = 2
       local input = torch.randn(batch, from, inj, ini)

       -- SpatialConvolutionCUDA requires tensors in weird shapes
       if testModule == nn.SpatialConvolutionCUDA then
          splitDim = 4
          concatDim = 1
          input = torch.randn(from, ini, inj, batch)
       end

       local input_cuda = input:cuda()

       local container = nn.ModelParallel(concatDim)
       local to_sum = 0
       for i=1,#to do
          to_sum = to_sum + to[i]
       end

       local module_correct = testModule(from, to_sum, ki, kj, si, sj)
       local offset = 1
       for i=1,#to do
          local module = testModule(from, to[i], ki, kj, si, sj);
          module.gradWeight:copy(module_correct.gradWeight:narrow(splitDim, offset, to[i]))
          module.gradBias:copy(module_correct.gradBias:narrow(1, offset, to[i]))
          module.weight:copy(module_correct.weight:narrow(splitDim, offset, to[i]))
          module.bias:copy(module_correct.bias:narrow(1, offset, to[i]))
          offset = offset + to[i]
          container:add(module)
       end

       module_correct:zeroGradParameters()
       container:zeroGradParameters()

       module_correct:cuda()
       container:cuda()

       -- forward pass
       local tm = {}
       times[name .. "_" .. "forward"] = tm
       tm.gpu = utils.timedRun(module_correct.forward, module_correct, input_cuda)
       tm.Ngpu = utils.timedRun(container.forward, container, input_cuda)

       --tm.multi_gpu = tic:time().real
       local ferr = module_correct.output:float() - container.output:float()
       ferr = ferr:abs():max()
       mytester:assertlt(ferr, precision, 'forward err')

       --backward pass
       local gradOutput = torch.randn(module_correct.output:size())
       local gradOutput_cuda = gradOutput:cuda()

       tm = {}
       times[name .. "_" .. "updateGradInput"] = tm
       tm.gpu = utils.timedRun(module_correct.updateGradInput, module_correct, input_cuda, gradOutput_cuda)
       tm.Ngpu = utils.timedRun(container.updateGradInput, container, input_cuda, gradOutput_cuda)
       local berr = module_correct.gradInput:float() - container.gradInput:float()
       berr = berr:abs():max()
       mytester:assertlt(berr, precision, 'backward err')

       tm = {}
       times[name .. "_" .. "accGradParameters"] = tm
       tm.gpu = utils.timedRun(module_correct.accGradParameters, module_correct, input_cuda, gradOutput_cuda, scale)
       tm.Ngpu = utils.timedRun(container.accGradParameters, container, input_cuda, gradOutput_cuda, scale)

       -- update params tests
       module_correct:updateParameters(learning_rate)
       container:updateParameters(learning_rate)

       local offset = 1
       for i, module in ipairs(container.modules) do
          local err = module.gradWeight:float() - module_correct.gradWeight:narrow(splitDim, offset, to[i]):float()
          mytester:assertlt(err:abs():max(), precision, 'gradWeight update error module#' .. i)

          local err = module.gradBias:float() - module_correct.gradBias:narrow(1, offset, to[i]):float()
          mytester:assertlt(err:abs():max(), precision, 'gradBias update error module#' .. i)

          local err = module.weight:float() - module_correct.weight:narrow(splitDim, offset, to[i]):float()
          mytester:assertlt(err:abs():max(), precision, 'weight update error module#' .. i)

          local err = module.bias:float() - module_correct.bias:narrow(1, offset, to[i]):float()
          mytester:assertlt(err:abs():max(), precision, 'bias update error module#' .. i)

          offset = offset + to[i]
       end
    end

    local function addSpatialConvolutionModules()
       testModules = {}
       testModules.SpatialConvolutionMM = nn.SpatialConvolutionMM
       testModules.SpatialConvolutionCUDA = nn.SpatialConvolutionCUDA
       for name, testModule in pairs(testModules) do
          mptest[name] = function()
              testSpatialConvolution(testModule, name)
          end
       end
    end

    utils.initializeGPUs()
    addSpatialConvolutionModules()

    local ordered_times = {}
    for n in pairs(times) do
       table.insert(ordered_times, n)
    end
    table.sort(ordered_times)

    for i, name in ipairs(ordered_times) do
       local time = times[name]
       print(string.format("%-40s %5.5fs (1-GPU) %5.5fs (%d-GPUs) x%2.2f speedup", name, time.gpu, time.Ngpu, nGPUs, time.gpu/time.Ngpu))
    end
end

LuaUnit:main()

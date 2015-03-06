-- Copyright 2004-present Facebook. All Rights Reserved.

require 'optim'
require 'cunn'
require 'fbcunn'  -- For nn.DataParallel
require 'fbnn'  -- For nn.Optim

local base_gpu = 1  -- Primary GPU to use
local num_gpus = 2  -- We will use {base_gpu, base_gpu+1, etc} with modulus
torch.setdefaulttensortype('torch.DoubleTensor')
torch.setnumthreads(8)
cutorch.setDevice(base_gpu)

-- Create an instance of the test framework
local precision = 5e-4
local mytester = torch.Tester()
local test = {}

function copyTable(x)  -- Shallow copy
  local ret = {}
  for k,v in pairs(x) do ret[k] = v end
  return ret
end

-- Build a dummy binary classifier. We will split the BATCHES across GPUs.
function buildNet(width, height, pool, feat, filt, num_convs)
  local net = nn.Sequential()
  assert(math.fmod(filt,2) == 1)
  for i = 1, num_convs do
    local fin = 3
    if (i > 1) then fin = feat end
    net:add(nn.SpatialConvolutionMM(fin, feat, filt, filt, 1, 1, (filt-1)/2))
    net:add(nn.Threshold())
  end
  net:add(nn.SpatialMaxPooling(pool, pool))
  net:add(nn.Reshape(width * height * feat / (pool * pool)))
  net:add(nn.Linear(width * height * feat / (pool * pool), 2))
  -- net:add(nn.SoftMax())  -- This is fake anyway, so just do regression :-)
  return net
end

function test.DataParallel()
  collectgarbage()
  local width = 16
  local height = 16
  local pool = 4
  local feat = 8
  local filt = 5
  local num_convs = 2
  local num_sgd_steps = 2
  local sync_gpu_cpu_params_every = 1
  local batch_size = 2 * num_gpus
  
  -- Build a CPU model
  local cpu_net = buildNet(width, height, pool, feat, filt, num_convs)

  -- Build a multi-GPU model
  local gpu_net = nn.DataParallel(1):cuda()
  for i = 1, num_gpus do
    local cur_gpu = math.fmod(base_gpu + (i-1)-1, cutorch.getDeviceCount())+1
    cutorch.setDevice(cur_gpu)
    gpu_net:add(cpu_net:clone():cuda(), cur_gpu)
  end
  cutorch.setDevice(base_gpu)

  local cpu_input = torch.rand(batch_size, 3, height, width)
  local gpu_input = cpu_input:cuda()
  local cpu_target = torch.rand(batch_size, 2)
  local gpu_target = cpu_target:cuda()   
 
  -- Set up an MSE optimizer on the GPU and CPU
  local optim_state_cpu = {
    learningRate = 1,  -- Artificially big learning rate
    weightDecay = 0,
  }
  local optim_state_gpu = copyTable(optim_state_cpu)
  local opt_cpu = nn.Optim(cpu_net, optim_state_cpu)
  local opt_gpu = nn.Optim(gpu_net, optim_state_gpu)
 
  local criterion_cpu = nn.MSECriterion()
  local criterion_gpu = criterion_cpu:clone():cuda()
  
  for i = 1, num_sgd_steps do
    collectgarbage()
     
    -- Perform an SGD step on the GPU and CPU
    opt_cpu:optimize(optim.sgd, cpu_input, cpu_target, criterion_cpu)
    opt_gpu:optimize(optim.sgd, gpu_input, gpu_target, criterion_gpu)
    assert(cutorch.getDevice() == base_gpu, 
      'DataParallel didnt restore GPU state to base_gpu')
    
    -- Now make sure that everything is the same
    local cpu_output = cpu_net.output
    local gpu_output = gpu_net.output
    local cpu_gradInput = cpu_net.gradInput
    local gpu_gradInput = gpu_net.gradInput
    local cpu_params, cpu_gradParams = cpu_net:parameters()
    local gpu_params, gpu_gradParams = gpu_net:get(1):parameters()

    mytester:assertlt((cpu_output - gpu_output:double()):abs():max(), 
      precision, 'fprop error ')
    mytester:assertlt((criterion_cpu.gradInput - 
      criterion_gpu.gradInput:double()):abs():max(), precision, 
      'CRITERION BPROP error ')
    mytester:asserteq(#cpu_params, #gpu_params)
    for j = 1, #cpu_params do
      mytester:assertlt((cpu_params[j] - gpu_params[j]:double()):abs():max(),
        precision, 'parameters error ')
    end
    mytester:asserteq(#cpu_gradParams, #gpu_gradParams)
    for j = 1, #cpu_gradParams do
      mytester:assertlt((cpu_gradParams[j] - 
        gpu_gradParams[j]:double()):abs():max(), precision, 
        'BPROP error (gradParams)')
    end
    mytester:assertlt((cpu_gradInput - gpu_gradInput:double()):abs():max(),
      precision, 'BPROP error (gradInput)')
    
    -- Sync the CPU and GPU weights every few "epochs" to prevent floating point
    -- drift between SGD iterations (ie, they will eventually be divergent after
    -- enough iterations)
    if math.fmod(i, sync_gpu_cpu_params_every) == 0 then
      for j = 1, #cpu_gradParams do
        cpu_params[j]:copy(gpu_params[j])
      end
    end
  end
end

-- Now run the test above
mytester:add(test)
mytester:run()

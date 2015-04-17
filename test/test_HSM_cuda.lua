-- Copyright 2004-present Facebook. All Rights Reserved.
-- Author: Michael Mathieu <myrhev@fb.com>

require 'cutorch'
require 'fbcunn'
require 'fbnn'

cutorch.setDevice(4)

print("Timing")
local n_iter = 100
for i = 1,1 do
    local n_clusters = 100
    local n_class = 10000
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
    local input_size = 1000
    local batch_size = 64
    local model_cpu = nn.HSM(mapping, input_size)
    local model_gpu = model_cpu:clone():cuda()

    local input_cpu = torch.randn(batch_size, input_size)
    local input_gpu = input_cpu:cuda()
    local target_cpu = torch.LongTensor(batch_size)
    for i = 1,batch_size do
        target_cpu[i] = torch.random(n_class)
    end
    local target_gpu = target_cpu:float():cuda()

    print("fprop")
    local t = torch.tic()
    for i = 1,n_iter do
        local loss_cpu, n_cpu = model_cpu:updateOutput(input_cpu, target_cpu)
    end
    print("cpu", torch.toc(t))
    t = torch.tic()
    cutorch.synchronize()
    for i = 1,n_iter do
        local loss_gpu, n_gpu = model_gpu:updateOutput(input_gpu, target_gpu)
    end
    cutorch.synchronize()
    print("gpu", torch.toc(t))

    print("bprop")
    local t = torch.tic()
    for i = 1,n_iter do
        local gradInput_cpu = model_cpu:updateGradInput(input_cpu, target_cpu)
    end
    print("cpu", torch.toc(t))
    t = torch.tic()
    cutorch.synchronize()
    for i = 1,n_iter do
        local gradInput_gpu = model_gpu:updateGradInput(input_gpu, target_gpu)
    end
    cutorch.synchronize()
    print("gpu", torch.toc(t))

    print("gradAcc")
    local t = torch.tic()
    for i = 1,n_iter do
        model_cpu:accGradParameters(input_cpu, target_cpu)
    end
    print("cpu", torch.toc(t))
    t = torch.tic()
    cutorch.synchronize()
    for i = 1,n_iter do
        model_gpu:accGradParameters(input_gpu, target_gpu)
    end
    cutorch.synchronize()
    print("gpu", torch.toc(t))
end

print("Correctness")
--torch.manualSeed(1)

for i = 1, 100 do
    print("Iteration " .. i)
    local n_clusters = torch.random(300)
    local n_class = torch.random(30000)
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
    local input_size = torch.random(3000) + 1
    local batch_size = torch.random(300)

    --print(n_clusters, n_class, input_size, batch_size)

    local model_cpu = nn.HSM(mapping, input_size)
    local model_gpu = model_cpu:clone():cuda()

    local input_cpu = torch.randn(batch_size, input_size)
    local input_gpu = input_cpu:cuda()
    local target_cpu = torch.Tensor(batch_size):fill(4)
    for i = 1,batch_size do
        target_cpu[i] = torch.random(n_class)
    end
    local target_gpu = target_cpu:cuda()

    local loss_cpu, n_cpu = model_cpu:updateOutput(input_cpu, target_cpu:long())
    local loss_gpu, n_gpu = model_gpu:updateOutput(input_gpu, target_gpu)

    function printdiff(cpu, gpu)
        local m = (cpu-gpu:double()):abs():max()
        local n = cpu:norm()
        print(m, n)
    end
    --printdiff(model_cpu.class_score, model_gpu.class_score)
    --printdiff(model_cpu.class_logsum, model_gpu.class_logsum)

    function assertdiff(cpu, gpu)
        if type(cpu) == 'number' then
            assert(math.abs(cpu-gpu) / math.abs(cpu) < 1e-3)
        else
            local m = (cpu-gpu:double()):abs():max()
            local n = cpu:norm()
            assert(m / n < 1e-3)
        end
    end

    --print (math.abs(loss_cpu - loss_gpu[1]), math.abs(loss_cpu))
    assertdiff(loss_cpu, loss_gpu[1])

    local gradInput_cpu = model_cpu:updateGradInput(input_cpu,
                                                    target_cpu:long())
    local gradInput_gpu = model_gpu:updateGradInput(input_gpu, target_gpu)

    local w_cpu, dw_cpu = model_cpu:getParameters()
    local w_gpu, dw_gpu = model_gpu:getParameters()

    --printdiff(gradInput_cpu, gradInput_gpu)
    assertdiff(gradInput_cpu, gradInput_gpu)

    model_cpu:zeroGradParameters()
    model_gpu:zeroGradParameters()
    model_cpu:accGradParameters(input_cpu, target_cpu:long())
    model_gpu:accGradParameters(input_gpu, target_gpu)

    --printdiff(dw_cpu, dw_gpu)
    assertdiff(dw_cpu, dw_gpu)

    -- test directUpdate
    local w, dw = model_gpu:getParameters()
    dw:normal()
    local initdw = dw:clone()
    local scale = torch.rand(1)[1]
    model_gpu:updateOutput(input_gpu, target_gpu)
    model_gpu:updateGradInput(input_gpu, target_gpu)
    model_gpu:accGradParameters(input_gpu, target_gpu, scale, false)
    local w1 = w:clone():add(dw)
    model_gpu:updateOutput(input_gpu, target_gpu)
    model_gpu:updateGradInput(input_gpu, target_gpu)
    model_gpu:accGradParameters(input_gpu, target_gpu, scale, true)
    w:add(initdw)
    local err = w:add(-1, w1):abs():max()
    assert(err < 1e-5)
end

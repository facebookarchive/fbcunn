require 'fbcunn'
require 'cudnn'

basenet=nn.Sequential()
layer=cudnn.SpatialConvolution(3,16,5,5,1,1)
basenet:add(layer)
basenet:add(cudnn.ReLU())

model=nn.DataParallel(1)
model:add(basenet)
model:add(basenet:clone())
model:cuda()

inputTensor=torch.CudaTensor(32,3,10,10)
model:forward(inputTensor)

foo=model.output:clone()
model:backward(inputTensor, foo)
print(inputTensor:size())
print(model.gradInput:size())

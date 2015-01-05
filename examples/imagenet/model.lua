--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
require 'nn'
require 'cunn'
require 'cudnn'
require 'fbcunn'
require 'optim'

--[[
   1. Create Model
     1.1. Create Alexnet convolutions
     1.3. Create Alexnet Classifier (fully connected layers)
     1.4. Combine 1.2 and 1.3 to produce final model
   2. Create Criterion
   3. If preloading option is set, preload weights from existing models appropriately
   4. Convert model to CUDA
]]--

-- 1.1. Create AlexNet
local features = nn.ModelParallel(2)

local fb1 = nn.Sequential() -- branch 1
fb1:add(cudnn.SpatialConvolution(3,48,11,11,4,4,2,2))       -- 224 -> 55
fb1:add(nn.ReLU())
fb1:add(nn.SpatialMaxPooling(3,3,2,2))                   -- 55 ->  27
fb1:add(nn.SpatialZeroPadding(2,2,2,2))
fb1:add(nn.SpatialConvolutionCuFFT(48,128,5,5,1,1))       --  27 -> 27
fb1:add(nn.ReLU())
fb1:add(nn.SpatialMaxPooling(3,3,2,2))                   --  27 ->  13
fb1:add(nn.SpatialZeroPadding(1,1,1,1))
fb1:add(nn.SpatialConvolutionCuFFT(128,192,3,3,1,1))      --  13 ->  13
fb1:add(nn.ReLU())
fb1:add(nn.SpatialZeroPadding(1,1,1,1))
fb1:add(nn.SpatialConvolutionCuFFT(192,192,3,3,1,1))      --  13 ->  13
fb1:add(nn.ReLU())
fb1:add(nn.SpatialZeroPadding(1,1,1,1))
fb1:add(nn.SpatialConvolutionCuFFT(192,128,3,3,1,1))      --  13 ->  13
fb1:add(nn.ReLU())
fb1:add(nn.SpatialMaxPooling(3,3,2,2))                   -- 13 -> 6

local fb2 = fb1:clone() -- branch 2
for k,v in ipairs(fb2:findModules('nn.SpatialConvolutionCuFFT')) do
   v:reset() -- reset branch 2's weights
end
for k,v in ipairs(fb2:findModules('nn.SpatialConvolution')) do
   v:reset() -- reset branch 2's weights
end

features:add(fb1)
features:add(fb2)

-- 1.3. Create Classifier (fully connected layers)
dropouts = {}
for i=1,4 do
   table.insert(dropouts, nn.Dropout(0.5))
end

local classifier = nn.Sequential()
classifier:add(nn.View(256*6*6))
classifier:add(dropouts[1])
classifier:add(nn.Linear(256*6*6, 4096))
classifier:add(nn.Threshold(0, 1e-6))
classifier:add(dropouts[2])
classifier:add(nn.Linear(4096, 4096))
classifier:add(nn.Threshold(0, 1e-6))
classifier:add(nn.Linear(4096, nClasses))
classifier:add(nn.LogSoftMax())

-- 1.4. Combine 1.1 and 1.3 to produce final model
model = nn.Sequential():add(features):add(classifier)

-- 2. Create Criterion
criterion = nn.ClassNLLCriterion()

print('=> Model')
print(model)

print('=> Criterion')
print(criterion)

-- 3. If preloading option is set, preload weights from existing models appropriately
if opt.retrain ~= 'none' then
   assert(paths.filep(opt.retrain), 'File not found: ' .. opt.retrain)
   print('Loading model from file: ' .. opt.retrain);
   model = torch.load(opt.retrain)
end

-- 4. Convert model to CUDA
print('==> Converting model to CUDA')
model = model:cuda()
criterion:cuda()

collectgarbage()

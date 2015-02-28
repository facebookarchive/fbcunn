--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
require 'torch'
require 'cutorch'
require 'paths'
require 'xlua'
require 'optim'

local opts = paths.dofile('opts.lua')

opt = opts.parse(arg)
print(opt)

torch.setdefaulttensortype('torch.FloatTensor')

cutorch.setDevice(opt.GPU) -- by default, use GPU 1
torch.manualSeed(opt.manualSeed)

print('Saving everything to: ' .. opt.save)
os.execute('mkdir -p ' .. opt.save)

paths.dofile('data.lua')
paths.dofile('model.lua')
paths.dofile('train.lua')
paths.dofile('test.lua')
paths.dofile('util.lua')

epoch = opt.epochNumber

for i=1,opt.nEpochs do
   train()
   test()
   epoch = epoch + 1
end

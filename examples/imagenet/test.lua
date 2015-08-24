--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))

local testDataIterator = function()
   testLoader:reset()
   return function() return testLoader:get_batch(false) end
end

local batchNumber
local top1_center, top5_center, loss
local top1_10crop, top5_10crop
local timer = torch.Timer()

function test()
   print('==> doing epoch on validation data:')
   print("==> online epoch # " .. epoch)

   batchNumber = 0
   cutorch.synchronize()
   timer:reset()

   -- set the dropouts to evaluate mode
   model:evaluate()

   top1_center = 0; top5_center = 0
   top1_10crop = 0; top5_10crop = 0
   loss = 0
   for i=1,math.ceil(nTest/opt.testBatchSize) do -- nTest is set in 1_data.lua
      local indexStart = (i-1) * opt.testBatchSize + 1
      local indexEnd = math.min(nImgs, indexStart + opt.testBatchSize - 1)
      donkeys:addjob(
         -- work to be done by donkey thread
         function()
            local inputs, labels = testLoader:get(indexStart, indexEnd)
            return sendTensor(inputs), sendTensor(labels)
         end,
         -- callback that is run in the main thread once the work is done
         testBatch
      )
      if i % 5 == 0 then
         donkeys:synchronize()
         collectgarbage()
      end
   end

   donkeys:synchronize()
   cutorch.synchronize()

   top1_center = top1_center * 100 / nTest
   top5_center = top5_center * 100 / nTest
   top1_10crop = top1_10crop * 100 / nTest
   top5_10crop = top5_10crop * 100 / nTest
   loss = loss / (nTest/opt.testBatchSize) -- because loss is calculated per batch
   testLogger:add{
      ['% top1 accuracy (test set) (center crop)'] = top1_center,
      ['% top5 accuracy (test set) (center crop)'] = top5_center,
      ['% top1 accuracy (test set) (10 crops)'] = top1_10crop,
      ['% top5 accuracy (test set) (10 crops)'] = top5_10crop,
      ['avg loss (test set)'] = loss
   }
   print(string.format('Epoch: [%d][TESTING SUMMARY] Total Time(s): %.2f \t'
                          .. 'average loss (per batch): %.2f \t '
                          .. 'accuracy [Center](%%):\t top-1 %.2f\t top-5 %.2f\t'
                          .. '[10crop](%%):\t top-1 %.2f\t top-5 %.2f',
                       epoch, timer:time().real, loss, top1_center, top5_center,
                       top1_10crop, top5_10crop))

   print('\n')


end -- of test()
-----------------------------------------------------------------------------
local inputsCPU = torch.FloatTensor()
local labelsCPU = torch.LongTensor()
local inputs = torch.CudaTensor()
local labels = torch.CudaTensor()

function testBatch(inputsThread, labelsThread)
   batchNumber = batchNumber + opt.testBatchSize

   receiveTensor(inputsThread, inputsCPU)
   receiveTensor(labelsThread, labelsCPU)
   inputs:resize(inputsCPU:size()):copy(inputsCPU)
   labels:resize(labelsCPU:size()):copy(labelsCPU)


   local outputs = model:forward(inputs)
   local err = criterion:forward(outputs, labels)
   cutorch.synchronize()
   local pred = outputs:float()

   loss = loss + err

   local function topstats(p, g)
      local top1 = 0; local top5 = 0
      _,p = p:sort(1, true)
      if p[1] == g then
          top1 = top1 + 1
          top5 = top5 + 1
      else
          for j=2,5 do
              if p[j] == g then
                  top5 = top5 + 1
                  break
              end
          end
      end
      return top1, top5, p[1]
  end

   for i=1,pred:size(1),10 do
      local p = pred[{{i,i+9},{}}]
      local g = labelsCPU[i]
      for j=0,9 do assert(labelsCPU[i] == labelsCPU[i+j]) end
      -- center
      local center = p[1] + p[2]
      local top1,top5 = topstats(center, g)
      top1_center = top1_center + top1
      top5_center = top5_center + top5
      -- 10crop
      local tencrop = p:sum(1)[1]
      local top1,top5,ans = topstats(tencrop, g)
      top1_10crop = top1_10crop + top1
      top5_10crop = top5_10crop + top5
   end
   if batchNumber % 64 == 0 then
      print(('Epoch: Testing [%d][%d/%d]'):format(
               epoch, batchNumber, nTest))
   end
end

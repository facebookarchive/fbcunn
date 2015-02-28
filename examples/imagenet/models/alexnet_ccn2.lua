require 'ccn2'
require 'cudnn'
function createModel(nGPU)
   local features = nn.Sequential() -- branch 1
   features:add(nn.Transpose({1,4},{1,3},{1,2}))
   features:add(ccn2.SpatialConvolution(3,64,11,4,0,1,4))       -- 224 -> 55
   features:add(cudnn.ReLU(true))
   features:add(ccn2.SpatialMaxPooling(3,2))                   -- 55 ->  27
   features:add(ccn2.SpatialConvolution(64,192,5,1,2,1,3))       --  27 -> 27
   features:add(cudnn.ReLU(true))
   features:add(ccn2.SpatialMaxPooling(3,2))                   --  27 ->  13
   features:add(ccn2.SpatialConvolution(192,384,3,1,1,1,3))      --  13 ->  13
   features:add(cudnn.ReLU(true))
   features:add(ccn2.SpatialConvolution(384,256,3,1,1,1,3))      --  13 ->  13
   features:add(cudnn.ReLU(true))
   features:add(ccn2.SpatialConvolution(256,256,3,1,1,1,3))      --  13 ->  13
   features:add(cudnn.ReLU(true))
   features:add(ccn2.SpatialMaxPooling(3,2))                   -- 13 -> 6
   features:add(nn.Transpose({4,1},{4,2},{4,3}))

   -- 1.3. Create Classifier (fully connected layers)
   local classifier = nn.Sequential()
   classifier:add(nn.View(256*6*6))
   classifier:add(nn.Dropout(0.5))
   classifier:add(nn.Linear(256*6*6, 4096))
   classifier:add(nn.Threshold(0, 1e-6))
   classifier:add(nn.Dropout(0.5))
   classifier:add(nn.Linear(4096, 4096))
   classifier:add(nn.Threshold(0, 1e-6))
   classifier:add(nn.Linear(4096, nClasses))
   classifier:add(nn.LogSoftMax())

   -- 1.4. Combine 1.1 and 1.3 to produce final model
   if nGPU > 1 then
      assert(nGPU <= cutorch.getDeviceCount(), 'number of GPUs less than nGPU specified')
      local features_single = features
      require 'fbcunn'
      features = nn.DataParallel(1)
      for i=1,nGPU do
         cutorch.withDevice(i, function()
                               features:add(features_single:clone())
         end)
      end
   end

   local model = nn.Sequential():add(features):add(classifier)

   return model
end

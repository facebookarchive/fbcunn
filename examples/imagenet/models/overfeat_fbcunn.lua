require 'fbcunn'
require 'cudnn'

function createModel(nGPU)
   local features = nn.Sequential()

   features:add(cudnn.SpatialConvolution(3, 96, 11, 11, 4, 4))
   features:add(cudnn.ReLU(true))
   features:add(cudnn.SpatialMaxPooling(2, 2, 2, 2))

   features:add(nn.SpatialConvolutionCuFFT(96, 256, 5, 5, 1, 1))
   features:add(cudnn.ReLU(true))
   features:add(cudnn.SpatialMaxPooling(2, 2, 2, 2))

   features:add(nn.SpatialZeroPadding(1,1,1,1))
   features:add(nn.SpatialConvolutionCuFFT(256, 512, 3, 3, 1, 1))
   features:add(cudnn.ReLU(true))

   features:add(nn.SpatialZeroPadding(1,1,1,1))
   features:add(nn.SpatialConvolutionCuFFT(512, 1024, 3, 3, 1, 1))
   features:add(cudnn.ReLU(true))

   features:add(cudnn.SpatialConvolution(1024, 1024, 3, 3, 1, 1, 1, 1))
   features:add(cudnn.ReLU(true))
   features:add(cudnn.SpatialMaxPooling(2, 2, 2, 2))

   -- 1.3. Create Classifier (fully connected layers)
   local classifier = nn.Sequential()
   classifier:add(nn.View(1024*5*5))
   classifier:add(nn.Dropout(0.5))
   classifier:add(nn.Linear(1024*5*5, 3072))
   classifier:add(nn.Threshold(0, 1e-6))

   classifier:add(nn.Dropout(0.5))
   classifier:add(nn.Linear(3072, 4096))
   classifier:add(nn.Threshold(0, 1e-6))

   classifier:add(nn.Linear(4096, nClasses))
   classifier:add(nn.LogSoftMax())

   -- 1.4. Combine 1.2 and 1.3 to produce final model
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

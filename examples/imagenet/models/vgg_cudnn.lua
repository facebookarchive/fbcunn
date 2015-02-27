function createModel(nGPU)
   require 'cudnn'

   local modelType = 'A' -- on a titan black, B/D/E run out of memory even for batch-size 32

   -- Create tables describing VGG configurations A, B, D, E
   local cfg = {}
   if modelType == 'A' then
      cfg = {64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'}
   elseif modelType == 'B' then
      cfg = {64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'}
   elseif modelType == 'D' then
      cfg = {64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'}
   elseif modelType == 'E' then
      cfg = {64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'}
   else
      error('Unknown model type: ' .. modelType .. ' | Please specify a modelType A or B or D or E')
   end

   local features = nn.Sequential()
   do
      local iChannels = 3;
      for k,v in ipairs(cfg) do
         if v == 'M' then
            features:add(cudnn.SpatialMaxPooling(2,2,2,2))
         else
            local oChannels = v;
            local conv3 = cudnn.SpatialConvolution(iChannels,oChannels,3,3,1,1,1,1);
            features:add(conv3)
            features:add(cudnn.ReLU(true))
            iChannels = oChannels;
         end
      end
   end

   local classifier = nn.Sequential()
   classifier:add(nn.View(512*7*7))
   classifier:add(nn.Linear(512*7*7, 4096))
   classifier:add(nn.Threshold(0, 1e-6))
   classifier:add(nn.Dropout(0.5))
   classifier:add(nn.Linear(4096, 4096))
   classifier:add(nn.Threshold(0, 1e-6))
   classifier:add(nn.Dropout(0.5))
   classifier:add(nn.Linear(4096, 1000))
   classifier:add(nn.LogSoftMax())

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

   local model = nn.Sequential()
   model:add(features):add(classifier)

   return model
end

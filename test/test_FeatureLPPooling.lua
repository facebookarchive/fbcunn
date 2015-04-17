-- Copyright 2004-present Facebook. All Rights Reserved.
require('fb.luaunit')
require('fbcunn')

local num_tries = 2
local jacobian = nn.Jacobian
local precision = 4e-3

local batch_max = 3
local feature_max = 100
local dim1_max = 3
local dim2_max = 3

local function pickPow()
   local num = torch.random(4)
   if num == 1 then
      return 1
   else
      return (num - 1) * 2.0
   end
end

local function runFPropTest(dims, width, stride, pow, batch_mode)
   local pool = nn.FeatureLPPooling(width, stride, pow, batch_mode):cuda()

   local num_batch = torch.random(batch_max)
   local num_features = (torch.random(feature_max) - 1) * stride + width
   local num_dim1 = torch.random(dim1_max)
   local num_dim2 = torch.random(dim2_max)

   print('test on dim ' .. dims ..
            ' features ' .. num_features ..
            ' width ' .. width .. ' stride ' .. stride ..
            ' p ' .. pow .. ' bm ' .. (batch_mode and 1 or 0))

   local input = nil
   if dims == 1 then
      if batch_mode then
         input = torch.FloatTensor(num_batch, num_features)

         for i = 1, num_batch do
            for f = 1, num_features do
               input[i][f] = f - 1
            end
         end

      else
         input = torch.FloatTensor(num_features)

         for f = 1, num_features do
            input[f] = f - 1
         end

      end
   elseif dims == 2 then
      if batch_mode then
         input = torch.FloatTensor(num_batch, num_features, num_dim1)

         for i = 1, num_batch do
            for f = 1, num_features do
               for j = 1, num_dim1 do
                  input[i][f][j] = f - 1
               end
            end
         end

      else
         input = torch.FloatTensor(num_features, num_dim1)

         for f = 1, num_features do
            for j = 1, num_dim1 do
               input[f][j] = f - 1
            end
         end

      end
   elseif dims == 3 then
      if batch_mode then
         input = torch.FloatTensor(num_batch, num_features, num_dim1, num_dim2)

         for i = 1, num_batch do
            for f = 1, num_features do
               for j = 1, num_dim1 do
                  for k = 1, num_dim2 do
                     input[i][f][j][k] = f - 1
                  end
               end
            end
         end

      else
         input = torch.FloatTensor(num_features, num_dim1, num_dim2)

         for f = 1, num_features do
            for j = 1, num_dim1 do
               for k = 1, num_dim2 do
                  input[f][j][k] = f - 1
               end
            end
         end

      end
   end

   input = input:cuda()
   local output = pool:forward(input):float()

   -- Each output feature o(k) (k zero based) for L1 is:
   -- sum(i((k - 1) * s), i((k - 1) * s + 1), ..., i((k - 1) * s + w - 1))
   -- if i(x) = x, then: o(k) = w * (k - 1) * s + w * (w - 1) / 2
   -- For Lp (p != 1), just evaluate ourselves and compare

   local function verifyFeature(val, k, width, stride, pow)
      local sum_input = 0
      if pow == 1 then
         sum_input = width * (k - 1) * stride + width * (width - 1) / 2
      else
         for w = 0, width - 1 do
            sum_input = sum_input + math.pow((k - 1) * stride + w, pow)
         end
         sum_input = math.pow(sum_input, 1 / pow)
      end

      local diff = math.abs(val - sum_input)
      if (diff >= 1e-3) then
         print('failed on ' .. val .. ' ' .. sum_input)
         assertTrue(math.abs(val - sum_input) < 1e-3)
      end
   end

   if dims == 1 then
      if batch_mode then
         for i = 1, output:size(1) do
            for f = 1, output:size(2) do
               verifyFeature(output[i][f], f, width, stride, pow)
            end
         end

      else
         for f = 1, output:size(1) do
            verifyFeature(output[f], f, width, stride, pow)
         end

      end
   elseif dims == 2 then
      if batch_mode then
         for i = 1, output:size(1) do
            for f = 1, output:size(2) do
               for j = 1, output:size(3) do
                  verifyFeature(output[i][f][j], f, width, stride, pow)
               end
            end
         end

      else
         for f = 1, output:size(1) do
            for j = 1, output:size(2) do
               verifyFeature(output[f][j], f, width, stride, pow)
            end
         end

      end
   elseif dims == 3 then
      if batch_mode then
         for i = 1, output:size(1) do
            for f = 1, output:size(2) do
               for j = 1, output:size(3) do
                  for k = 1, output:size(4) do
                     verifyFeature(output[i][f][j][k], f, width, stride, pow)
                  end
               end
            end
         end

      else
         for f = 1, output:size(1) do
            for j = 1, output:size(2) do
               for k = 1, output:size(3) do
                  verifyFeature(output[f][j][k], f, width, stride, pow)
               end
            end
         end

      end
   end
end

local function runBPropTest(dims, width, stride, pow, batch_mode)
   local pool = nn.FeatureLPPooling(width, stride, pow, batch_mode):cuda()

   local num_batch = torch.random(batch_max)
   local num_features = (torch.random(feature_max) - 1) * stride + width
   local num_dim1 = torch.random(dim1_max)
   local num_dim2 = torch.random(dim2_max)

   local input = nil
   if dims == 1 then
      if batch_mode then
         input = torch.CudaTensor(num_batch, num_features)
      else
         input = torch.CudaTensor(num_features)
      end
   elseif dims == 2 then
      if batch_mode then
         input = torch.CudaTensor(num_batch, num_features, num_dim1)
      else
         input = torch.CudaTensor(num_features, num_dim1)
      end
   elseif dims == 3 then
      if batch_mode then
         input = torch.CudaTensor(num_batch, num_features, num_dim1, num_dim2)
      else
         input = torch.CudaTensor(num_features, num_dim1, num_dim2)
      end
   end

   local err = jacobian.testJacobian(pool, input, -2, -2, 5e-4)
   print('test on dim ' .. dims ..
            ' features ' .. num_features ..
            ' width ' .. width .. ' stride ' .. stride ..
            ' p ' .. pow .. ' err ' .. err)
   assertTrue(err < precision)
end

function testForwardLp()
   for i = 1, num_tries do
      for stride = 1, 4 do
         for idx, batch_mode in ipairs({true, false}) do
            for dims = 1, 3 do
               runFPropTest(dims, 1 + torch.random(15),
                            stride, pickPow(), batch_mode)
            end
         end
      end
   end
end

function testZeroBProp()
   local pool = nn.FeatureLPPooling(3, 1, 2.0, false):cuda()

   local input = torch.CudaTensor(100):zero()
   pool:forward(input)

   local gradOutput = torch.CudaTensor(98):zero()
   local gradInput = pool:backward(input, gradOutput, 1.0)

   for i = 1, gradInput:size(1) do
      assertTrue(gradInput[i] == 0)
   end
end

function testJacobian1dNoBatch()
   for i = 1, num_tries do
      for stride = 1, 4 do
         runBPropTest(1, 1 + torch.random(15), stride, pickPow(), false)
      end
   end
end

function testJacobian1dBatch()
   for i = 1, num_tries do
      for stride = 1, 4 do
         runBPropTest(1, 1 + torch.random(15), stride, pickPow(), true)
      end
   end
end

function testJacobian2dNoBatch()
   for i = 1, num_tries do
      for stride = 1, 4 do
         runBPropTest(2, 1 + torch.random(15), stride, pickPow(), false)
      end
   end
end

function testJacobian2dBatch()
   for i = 1, num_tries do
      for stride = 1, 4 do
         runBPropTest(2, 1 + torch.random(15), stride, pickPow(), true)
      end
   end
end

function testJacobian3dNoBatch()
   for i = 1, num_tries do
      for stride = 1, 4 do
         runBPropTest(3, 1 + torch.random(15), stride, pickPow(), false)
      end
   end
end

function testJacobian3dBatch()
   for i = 1, num_tries do
      for stride = 1, 4 do
         runBPropTest(3, 1 + torch.random(15), stride, pickPow(), true)
      end
   end
end

LuaUnit:main()

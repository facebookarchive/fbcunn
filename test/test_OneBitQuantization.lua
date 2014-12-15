-- Copyright 2004-present Facebook. All Rights Reserved.

require('fb.luaunit')
require('fbcunn')

local num_tries = 10

function testExactQuantization()
   for tries = 1, num_tries do
      local q = nn.OneBitQuantization()
      local t = torch.CudaTensor(torch.random(50), torch.random(50))

      print('Quantizing ' .. t:size(1) .. ' x ' .. t:size(2))

      -- We will get exact quantization if there is only one positive
      -- and one negative value in each row.
      for row = 1, t:size(1) do
         local pos_value = torch.uniform(10)
         local neg_value = -torch.uniform(10)

         for col = 1, t:size(2) do
            local val = pos_value

            if torch.bernoulli(0.5) == 0 then
               val = neg_value
            end

            t[row][col] = val
         end
      end

      local quantized = q:quantize(t)
      local dequantized =
         q:dequantize(quantized, q.avg_pos, q.avg_neg, t:size(2))

      assertTrue((dequantized:float() - t:float()):abs():max() < 1e-5)
   end
end

function testErrorDecaysToZero()
   for tries = 1, num_tries do
      -- In order to show that quantization error works, we should be
      -- able to send a matrix via quantization, and then successfully
      -- send the zero matrix.
      -- For each successive pass, the quantization error should diminish
      -- and on the receiving side, we should get something that approximates
      -- the original matrix.
      local q = nn.OneBitQuantization()

      -- Send two matrices
      local orig1 =
         torch.randn(10 + torch.random(30), 10 + torch.random(30)):cuda()
      local orig2 = torch.randn(orig1:size(1), orig1:size(2)):cuda()

      -- This is the signal that we wish to approximate
      local orig = orig1:float() + orig2:float()

      print('Quantizing ' .. orig:size(1) .. ' x ' .. orig:size(2))

      -- pass `orig1`
      local quantized = q:quantize(orig1)
      local dequantized =
         q:dequantize(quantized, q.avg_pos, q.avg_neg, orig1:size(2))

      -- dequantized will become the approximation to `orig`
      local approx = dequantized:float()

      -- pass `orig2`
      quantized = q:quantize(orig2)
      dequantized =
         q:dequantize(quantized, q.avg_pos, q.avg_neg, orig2:size(2))
      approx:add(dequantized:float())

      -- Now, after sending some signal, we will pass 0 a couple of times, in
      -- order to flush the quantization error. The number of passes required is
      -- related to the size of the original matrix and is also dependent upon
      -- floating point precision.
      local zeros = torch.CudaTensor(orig:size(1), orig:size(2))
      zeros:zero()

      for passes = 1, 100 do
         quantized = q:quantize(zeros)
         dequantized =
            q:dequantize(quantized, q.avg_pos, q.avg_neg, zeros:size(2))

         approx:add(dequantized:float())
      end

      assertTrue((orig:float() - approx):abs():max() < 5e-4)
   end
end

LuaUnit:main()

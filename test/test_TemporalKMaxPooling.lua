-- Copyright 2004-present Facebook. All Rights Reserved.

require('fb.luaunit')

require('math')
require('nn')
require('fbtorch')
require('fbcunn')

local tester = torch.Tester()
local TemporalKMaxPoolingTest = {}
local test_repeats = 5

local function runUpdateOutput(batch, n, d, k)
   -- batch = nil or size of batch
   -- n = number of words
   -- d = dimension of embeddings
   -- k = k-max pooling
   local input = nil
   if batch then
      input = torch.randn(batch, n, d):cuda()
   else
      input = torch.randn(n, d):cuda()
   end

   local kmax = nn.TemporalKMaxPooling(k)

   local output = kmax:updateOutput(input)
   assert(output == kmax.output)

   if batch then
      assert(kmax.output:size(2) == k)
      assert(kmax.output:size(3) == input:size(3))

      for i = 1, batch do
         local kth_max = torch.min(kmax.output[i]:double(), 1)
         local kth_max = kth_max:expand(input[i]:size(1), input[i]:size(2));

         local count_at_least_kth_max =
            torch.le(kth_max, input[i]:double()):sum(1)

         assert(torch.eq(count_at_least_kth_max, math.min(n, k)):sum() == d)
      end

   else
      assert(kmax.output:size(1) == k)
      assert(kmax.output:size(2) == input:size(2))

      local kth_max = torch.min(kmax.output:double(), 1)
      local kth_max = kth_max:expand(input:size(1), input:size(2));

      local count_at_least_kth_max = torch.le(kth_max, input:double()):sum(1)

      assert(torch.eq(count_at_least_kth_max, math.min(n, k)):sum() == d)
   end
end

local function runUpdateGradInput(batch, n, d, k)
   -- batch = nil or size of batch
   -- n = number of words
   -- d = dimension of embeddings
   -- k = k-max pooling
   -- infer_length = if true, have the module infer the length from the batch

   local input = nil
   if batch then
      input = torch.randn(batch, n, d):cuda()
   else
      input = torch.randn(n, d):cuda()
   end

   local kmax = nn.TemporalKMaxPooling(k)
   local output = kmax:updateOutput(input)
   local delta = torch.randn(output:size()):cuda()

   local gradInput = kmax:updateGradInput(input, delta)
   assert(gradInput == kmax.gradInput)

   if batch then
      assert(kmax.gradInput:size(2) == input:size(2))
      assert(kmax.gradInput:size(3) == input:size(3))

      for i = 1, batch do
         local grad_input_sum = torch.sum(kmax.gradInput[i]:double())
         local delta_sum = torch.sum(delta[i]:double()[{{1, math.min(n,k)}}])

         assert(math.abs(grad_input_sum - delta_sum) < 1e-6)
      end
   else
      assert(kmax.gradInput:size(1) == input:size(1))
      assert(kmax.gradInput:size(2) == input:size(2))

      local grad_input_sum = torch.sum(kmax.gradInput:double())
      local delta_sum = torch.sum(delta:double()[{{1, math.min(n,k)}}])

      assert(math.abs(grad_input_sum - delta_sum) < 1e-6)
   end
end

function TemporalKMaxPoolingTest.updateOutput()
   for i = 1, test_repeats do
      local n = torch.random(10, 100)
      local d = torch.random(10, 200)
      local k = torch.random(1, n)

      print('running for (' .. n .. ', ' .. d .. ') choose ' .. k)
      runUpdateOutput(nil, n, d, k)
   end
end

function TemporalKMaxPoolingTest.updateOutputBatch()
   for i = 1, test_repeats do
      local batch = torch.random(1, 40)
      local n = torch.random(10, 160)
      local d = torch.random(10, 200)
      local k = torch.random(1, n)

      print('running for (' .. batch .. ', ' .. n ..
               ', ' .. d .. ') choose ' .. k)
      runUpdateOutput(batch, n, d, k)
   end
end

function TemporalKMaxPoolingTest.updateGradInput()
   for i = 1, test_repeats do
      local n = torch.random(10, 160)
      local d = torch.random(10, 200)
      local k = torch.random(1, n)

      print('running for (' .. n .. ', ' .. d .. ') choose ' .. k)
      runUpdateGradInput(nil, n, d, k)
   end
end

function TemporalKMaxPoolingTest.updateGradInputBatch()
   for i = 1, test_repeats do
      local batch = torch.random(1, 40)
      local n = torch.random(10, 160)
      local d = torch.random(10, 200)
      local k = torch.random(1, n)

      print('running for (' .. batch .. ', ' .. n ..
               ', ' .. d .. ') choose ' .. k)
      runUpdateGradInput(batch, n, d, k)
   end
end

function TemporalKMaxPoolingTest.sequential()
   local input = torch.randn(10, 5):cuda()
   local seq = nn.Sequential()

   local kmax = nn.TemporalKMaxPooling(5)
   seq:add(nn.TemporalKMaxPooling(5))

   local kmax_output = kmax:updateOutput(input)
   local seq_output = seq:updateOutput(input)
   local output_matches = torch.eq(kmax_output:double(), seq_output:double())
   assert (output_matches:sum() == output_matches:numel())

   local delta = torch.randn(kmax.output:size()):cuda()

   local kmax_gradInput = kmax:updateGradInput(input, delta)
   local seq_gradInput = kmax:updateGradInput(input, delta)
   local gradInput_matches =
      torch.eq(kmax_gradInput:double(), seq_gradInput:double())
   assert (gradInput_matches:sum() == gradInput_matches:numel())
end

function TemporalKMaxPoolingTest.dynamic()
  local kmax = nn.TemporalKMaxPooling(2, 0.5)
  local seq = nn.Sequential()
  seq:add(nn.TemporalKMaxPooling(2, 0.5))

  for n=12,13 do
    local input = torch.randn(n, 1):cuda()
    local kmax_output = kmax:updateOutput(input)
    local seq_output = seq:updateOutput(input)
    assert (kmax_output:size(1) == 6)
    assert (torch.all(kmax_output:eq(seq_output)))
  end
end

tester:add(TemporalKMaxPoolingTest)
tester:run()

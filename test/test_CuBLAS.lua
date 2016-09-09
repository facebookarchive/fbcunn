-- Copyright 2004-present Facebook. All Rights Reserved.
require 'fb.luaunit'
require 'fbtorch'
require 'cunn'
require 'fbcunn'

torch.setdefaulttensortype('torch.FloatTensor')

local fb_test = {}

-- Let C = m-by-n and A = m-by-k
-- Format is m, n, k, seqIter, batch, numHandles, numStreams
local problemSize = {
-- Sanity tests
-- Trivial mxm, no batch, no iter
    {1, 1, 2, {}, {}, 0, 0},
    {1, 1, 2, {}, {}, 0, 1},
    {1, 1, 2, {}, {}, 1, 0},
    {1, 1, 2, {}, {}, 1, 1},
    {1, 1, 2, {}, {}, 16, 16},
-- 2x4 <- 2x8 * 8x4 as 1 iter, 1 batch
   {2, 4, 8, {1}, {1}, 1, 1},
-- 2x4 <- 2x8 * 8x4 as 1 iter, no batch
    {2, 4, 8, {1}, {}, 1, 1},
-- 2x4 <- 2x8 * 8x4 as no iter, 1 batch
    {2, 4, 8, {}, {1}, 1, 1},
-- 2x4 <- 2x8 * 8x4 as no iter, no batch
    {2, 4, 8, {}, {}, 1, 1},
-- 128x128 <- 128x128 * 128x128 as 4x4 iter, 4x4 batch
    {128, 128, 128, {4, 4}, {4, 4}, 1, 1},
    {1024, 1024, 1024, {1, 1}, {1, 1}, 1, 1},
    {1024, 1024, 1024, {}, {}, 1, 1},
--  Various way of performing temporal convolution of 512: 32 -> 16
    {16, 1024, 512, {}, {1}, 1, 1},
    {16, 1024, 512, {}, {}, 1, 1},
    {1, 1024, 512, {16}, {1}, 1, 1},
    {1, 1024, 512, {1}, {16}, 1, 1},
    {32 * 16, 1024, 512, {1}, {1}, 1, 1},
    {1, 1024, 512, {16 * 32}, {1}, 1, 1},
    {16, 1024, 512, {32}, {1}, 16, 1},
    {16, 1024, 512, {1}, {32}, 0, 0},
    {1, 1024, 512, {1}, {16 * 32}, 1, 1},
  }

-- This test exercises the performance of multi-handle + multi-stream on many
-- small gemms.
local _testMultiHandlePerf = {
  {513, 513, 513, {53}, {}, 0, 0},
  {513, 513, 513, {53}, {}, 1, 1},
  {513, 513, 513, {53}, {}, 1, 4},
  {513, 513, 513, {53}, {}, 4, 1},
  {513, 513, 513, {53}, {}, 4, 4},
}

local function concat(t1,t2)
    local res = {}
    for i=1,#t1 do
        res[#res + 1] = t1[i]
    end
    for i=1,#t2 do
        res[#res + 1] = t2[i]
    end
    return res
end

local function testLoop(problemSize)
  -- Just allocate some dummy placeholder to get to the proper
  -- function in the lua module
  local net = nn.CuBLASWrapper(true)

  local m = problemSize[1]
  local n = problemSize[2]
  local k = problemSize[3]
  local seqIter = problemSize[4]
  local batch = problemSize[5]
  local handles = problemSize[6]
  local streams = problemSize[7]
  local seqBatch = concat(seqIter, batch)
  local sA = torch.LongStorage(concat(seqBatch, {m, k}))
  local sB = torch.LongStorage(concat(seqBatch, {k, n}))
  local sC = torch.LongStorage(concat(seqBatch, {m, n}))
  local A = torch.Tensor(sA):cuda()
  local B = torch.Tensor(sB):cuda()
  local C = torch.Tensor(sC):cuda()

  cutorch.reserveBlasHandles(handles)
  cutorch.reserveStreams(streams)
  cutorch.synchronize()
  net:matmult(A, B, C, seqIter, batch)
  mytester:assert(true)

  cutorch.synchronize()
  collectgarbage()
end

function fb_test.testGEMMs()
  for i = 1, table.getn(_testMultiHandlePerf) do
    testLoop(_testMultiHandlePerf[i])
  end
  for i = 1, table.getn(problemSize) do
    testLoop(problemSize[i])
  end
end

mytester = torch.Tester()
mytester:add(fb_test)
mytester:run()

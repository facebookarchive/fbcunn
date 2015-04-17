-- Copyright 2004-present Facebook. All Rights Reserved.
require('fb.luaunit')

require 'cunn'

require 'fbcunn'

torch.setdefaulttensortype('torch.FloatTensor')

local test = {}

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

-- Soumith's inline print
local ndepth = 4
local function print_inline(...)
   local function rawprint(o)
      io.write(tostring(o or '') .. ' ')
      io.flush()
   end

   local function printrecursive(obj, depth)
      local depth = depth or 0
      local tab = 0
      local line = function(s) for i=1,tab do io.write(' ') end rawprint(s) end
         if next(obj) then
            line('{')
            for k,v in pairs(obj) do
               if type(v) == 'table' then
                  if depth >= (ndepth-1) or next(v) == nil then
                     line(tostring(k) .. ' : {}')
                  else
                     line(tostring(k) .. ' : ') printrecursive(v, depth + 1)
                  end
               else
                  line(tostring(k) .. ' : ' .. v)
               end
               rawprint(',')
            end
            tab = tab-2
            line('}')
         else
            line('{}')
         end
   end
   for i = 1,select('#',...) do
      local obj = select(i,...)
      if type(obj) ~= 'table' then
         if type(obj) == 'userdata' or type(obj) == 'cdata' then
            rawprint(obj)
         else
            io.write(obj .. '\t')
            if i == select('#',...) then
               rawprint()
            end
         end
      elseif getmetatable(obj) and getmetatable(obj).__tostring then
         rawprint(obj)
      else
         printrecursive(obj)
      end
   end
end

local function testLoop(problemSize)
  -- Just allocate some dummy placeholder to get to the proper
  -- function in the lua module
  local net = nn.CuBLASWrapper()

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

  print_inline(problemSize)
  print('')
  net:matmult(A, B, C, seqIter, batch, handles, streams)

  collectgarbage()
end

for i = 1, table.getn(problemSize) do
   testLoop(problemSize[i])
end

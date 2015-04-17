-- Copyright 2004-present Facebook. All Rights Reserved.
require('fb.luaunit')

require 'cunn'

require 'fbcunn'

torch.setdefaulttensortype('torch.FloatTensor')

local test = {}

-- These are used for fast, exhaustive search over the parameters space
-- Can be overridden by setting problemSizes to non-{}
local batchList = {
      128, 64, 32,
      }
local filterList = {
      128, 96, 64, 32, 24,
      }
local planeList = {
      128, 96, 64, 32, 24, 3
      }
local inputRowList = {
      128, 96, 64, 32, 16, 13
      }
local inputColList = {
      128, 96, 64, 32, 16, 13
      }
local kernelRowList = {
      11, 9, 7, 5, 3
      }
local kernelColList = {}

-- batch, filters, plane, row, col, kernelRow, kernelCol overrides
-- the List arguments
-- This is particularly useful to explore tradeoffs between cufft
-- efficiency at various interpolation sizes and amount of work in
-- transpose + mxm

-- Soumith's benchmark sizes
local fixedSizes = {
--    {128,  96,   3, 128, 128,  11,  11},
--    {128,  64,  64,  64,  64,   9,   9},
--    {128, 128, 128,  32,  32,   9,   9},
--    {128, 128, 128,  16,  16,   7,   7},
--    {128, 384, 384,  13,  13,   3,   3},
    {128, 96, 256,  31,  31,   5,   5}, -- 1 GPU
    {128, 96, 128,  31,  31,   5,   5}, -- 2 GPU
    {64, 96, 256,  31,  31,   5,   5},  -- 2 GPU
    {128, 96, 256, 21,  31,   5,   5},  -- 2 GPU, 27 / 2 = 14 + 4 + 3
    {128, 96, 64,  31,  31,   5,   5},  -- 4 GPU
    {32, 96, 256,  31,  31,   5,   5},  -- 4 GPU
    {128, 96, 256, 14,  31,   5,   5},  -- 4 GPU, 27 / 4 = 7 + 4 + 3
    {64, 96, 256, 21,  31,   5,   5},   -- 4 GPU, 27 / 2 = 14 + 4 + 3
    {128, 96, 128, 21,  31,   5,   5},  -- 4 GPU, 27 / 2 = 14 + 4 + 3
    {64, 96, 128,  31,  31,   5,   5},  -- 2 GPU
  }

-- Running         76      81      84      8       9       92      88
-- Running         176     3       9       8       1       13      54

-- Set this to {} to run a small search around the fixedSizes
local problemSizes = fixedSizes -- {}

local problemSize = {}

local function testLoop()
  -- Just allocate some dummy placeholder to get to the proper
  -- function in the lua module
  local net = nn.SpatialConvolutionCuFFT(1, 1, 1, 1)
  local input = torch.Tensor(1, 1, 1, 1):normal():cuda()

  if table.getn(problemSize) > 0 then
    batchList = {problemSize[1]}
    filterList = {problemSize[2]}
    planeList = {problemSize[3]}
    inputRowList = {problemSize[4]}
    inputColList = {problemSize[5]}
    kernelRowList = {problemSize[6]}
    kernelColList = {}
  end

  local batches = torch.Tensor(batchList):cuda()
  local filters = torch.Tensor(filterList):cuda()
  local planes = torch.Tensor(planeList):cuda()
  local inputRows = torch.Tensor(inputRowList):cuda()
  local inputCols = torch.Tensor(inputColList):cuda()
  local kernelRows = torch.Tensor(kernelRowList):cuda()
  local kernelCols = torch.Tensor(kernelColList):cuda()

  print('-------------------------------------------------------')
  net:explorePerformance(input, batches, filters,
    planes, inputRows, inputCols, kernelRows, kernelCols)

  net:cleanupBuffers(input)
  collectgarbage()
end

if table.getn(problemSizes) >= 1 then
  for i = 1, table.getn(problemSizes) do
    problemSize = problemSizes[i]
    testLoop()
  end
else
  testLoop()
end

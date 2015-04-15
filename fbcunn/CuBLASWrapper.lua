-- Copyright 2004-present Facebook. All Rights Reserved.

local CuBLASWrapper = torch.class('nn.CuBLASWrapper')

function CuBLASWrapper:__init()
   self.iterDims = 0
   self.batchDims = 0
   self.handles = 0
   self.streams = 0
end

function CuBLASWrapper:matmult(A, B, C, iterDims, batchDims, handles, streams)
   self.iterDims = table.getn(iterDims) or 0
   self.batchDims = table.getn(batchDims) or 0
   self.handles = handles or 0
   self.streams = streams or 0
   A.nn.CuBLASWrapper_matmult(self, A, B, C)
end

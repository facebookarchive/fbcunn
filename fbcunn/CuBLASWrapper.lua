-- Copyright 2004-present Facebook. All Rights Reserved.

local CuBLASWrapper = torch.class('nn.CuBLASWrapper')

function CuBLASWrapper:__init(timed)
   self.iterDims = 0
   self.batchDims = 0
   self.handles = 0
   self.streams = 0
   self.timed = timed or false
end

function CuBLASWrapper:matmult(
      A, B, C, iterDims, batchDims, transA, transB, scale)
   self.transA = transA or 'n'
   self.transB = transB or 'n'
   self.iterDims = table.getn(iterDims) or 0
   self.batchDims = table.getn(batchDims) or 0
   self.scale = scale or 1.0
   A.nn.CuBLASWrapper_matmult(self, A, B, C)
end

function CuBLASWrapper:matmultComplex(
      A, B, C, iterDims, batchDims, transA, transB, scale)
   self.transA = transA or 'n'
   self.transB = transB or 'n'
   self.iterDims = table.getn(iterDims) or 0
   self.batchDims = table.getn(batchDims) or 0
   self.scale = scale or 1.0
   A.nn.CuBLASWrapper_matmultComplex(self, A, B, C)
end

function CuBLASWrapper:transpose(
      A, B, separator, transposeMetaData, handle, stream)
   self.separator = separator or 0
   self.transposeMetaData = transposeMetaData or false
   self.handle = handle or 1 -- always handle 1 by default
   self.stream = stream or 0
   A.nn.CuBLASWrapper_transpose(self, A, B)
end

function CuBLASWrapper:transposeComplex(
      A, B, separator, transposeMetaData, handle, stream)
   self.separator = separator or 0
   self.transposeMetaData = transposeMetaData or false
   self.handle = handle or 1 -- always handle 1 by default
   self.stream = stream or 0
   A.nn.CuBLASWrapper_transposeComplex(self, A, B)
end

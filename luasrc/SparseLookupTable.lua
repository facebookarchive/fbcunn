-- Copyright 2004-present Facebook. All Rights Reserved.

--[[
Sparse lookup table. Similar to the regular LookupTable.lua module, 
except for the following differences:

1. The outputs are in sparse format.
2. The inputs are pairs (i,w), so the output corresponding to index i
is scaled by w.
3. The indices are fixed, i.e. during a parameter update only the nonzero 
coefficents are updated. This is to avoid having to create new indices, 
which is expensive and may result in the weights no longer being sparse.
]]
local SparseLookupTable, parent = torch.class('nn.SparseLookupTable','nn.Module')

sparse = require('sparse')

--[[
Parameters:
* `indices` is a 2D matrix of indices which will be nonzero.
* `sparseGrad` indicates whether incoming gradients will be sparse or dense.
]]
function SparseLookupTable:__init(indices,sparseGrad)
  parent.__init(self)

  self.nEntities = indices:size(1)
  self.nIndices = indices:size(2)
  self.sparseGrad = sparseGrad or true
  self.weight = torch.Tensor(self.nEntities,self.nIndices,2)
  self.weight[{{},{},1}]:copy(indices)
  self.gradWeight = torch.Tensor(self.nEntities,self.nIndices,2)
  self:reset()
end

function SparseLookupTable:reset(stdv)
  stdv = stdv or 1
  self.weight[{{},{},2}]:normal(0, stdv)
end

function SparseLookupTable:updateOutput(input)
  local nIndex = input:size(1)
  self.output:resize(nIndex,self.nIndices,2)
  for i=1,nIndex do
    local indx = input[i][1]
    local weight = input[i][2]
    self.output[i]:copy(self.weight[indx])
    self.output[i][{{},2}]:mul(weight)
  end
  return self.output
end

function SparseLookupTable:accUpdateGradParameters(input, gradOutput,lr)
  for i=1,input:size(1) do
    local indx = input[i][1]
    local weight = input[i][2]
    if self.sparseGrad then
      sparse.SaddSoverlap(self.weight[indx], gradOutput[i], -lr*weight)
    else
      sparse.SaddDoverlap(self.weight[indx], gradOutput[i], -lr*weight)
    end
  end
end

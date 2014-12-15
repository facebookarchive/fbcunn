-- Copyright 2004-present Facebook. All Rights Reserved.

-- Sum module for sparse vectors.
local SparseSum, parent = torch.class('nn.SparseSum','nn.Module')

function SparseSum:__init()
  parent.__init(self)
end

function SparseSum:updateOutput(input)
  local nInputs = input:size(1)
  if nInputs == 1 then
    self.output = input[1]:clone()
  else
    self.output = sparse.sumSameSizeSupport(input)
  end
  return self.output
end

function SparseSum:updateGradInput(input, gradOutput)
  self.gradInput:resize(input:size(1),gradOutput:size(1),2)
  for i = 1,input:size(1) do 
    self.gradInput[i]:copy(gradOutput)
  end
  return self.gradInput
end


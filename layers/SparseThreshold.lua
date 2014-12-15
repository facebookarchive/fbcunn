-- Copyright 2004-present Facebook. All Rights Reserved.

-- Same as Threshold module, for sparse vectors. 
local SparseThreshold, parent = torch.class('nn.SparseThreshold','nn.Module')

function SparseThreshold:__init(th,v)
  parent.__init(self)
  self.module = nn.Threshold(th,v)
end

function SparseThreshold:updateOutput(input)
  self.output:resize(input:size())
  local dim = input:nDimension()
  local input_indices = input:select(dim,1)
  local input_data = input:select(dim,2)
  local output_indices = self.output:select(dim,1)
  local output_data = self.output:select(dim,2)
  output_indices:copy(input_indices)
  output_data:copy(self.module:updateOutput(input_data))
  return self.output
end

function SparseThreshold:updateGradInput(input, gradOutput)
  self.gradInput:resize(input:size())
  local dim = input:nDimension()
  local input_data = input:select(dim,2)
  local gradInput_indices = self.gradInput:select(dim,1)
  local gradInput_data = self.gradInput:select(dim,2)
  local gradOutput_indices = gradOutput:select(dim,1)
  local gradOutput_data = gradOutput:select(dim,2)
  gradInput_indices:copy(gradOutput_indices)
  gradInput_data:copy(self.module:updateGradInput(input_data,gradOutput_data))
  return self.gradInput
end


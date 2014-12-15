-- Copyright 2004-present Facebook. All Rights Reserved.

local SparseConverter, parent = torch.class('nn.SparseConverter','nn.Module')



--[[
Parameters:
* `fconv` - conversion to perform in fprop, either 'StoD','DtoS' or nil
* `bconv` - conversion to perform in bprop, either 'StoD','DtoS' or nil
* `dim` - number of dimensions
* `thresh` - threshold for sparsifying (0 by default)
]]
function SparseConverter:__init(fconv,bconv,dim,thresh)
  parent.__init(self)
  if fconv == 'DtoS' and bconv == 'DtoS' 
    or fconv == 'StoD' and bconv == 'StoD' then 
    error('incompatible transformations') 
  end
  self.dim = dim
  self.fconv = fconv
  self.bconv = bconv
  self.thresh = thresh or 0
end

function SparseConverter:updateOutput(input)
  if self.fconv == 'StoD' then
    self.output = sparse.StoD(input,self.dim)
  elseif self.fconv == 'DtoS' then
    self.output = sparse.DtoS(input,self.thresh)
  else
    self.output = input
  end
  return self.output
end

function SparseConverter:updateGradInput(input, gradOutput)
  if self.bconv == 'StoD' then
    self.gradInput = sparse.StoD(gradOutput,self.dim)
  elseif self.bconv == 'DtoS' then
    self.gradInput = sparse.DtoS(gradOutput,self.thresh)
  else
    self.gradInput = gradOutput
  end
  return self.gradInput
end


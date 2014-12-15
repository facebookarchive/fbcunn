-- Copyright 2004-present Facebook. All Rights Reserved.

local sparse = require('sparse')
require('utils')

--[[
This module performs a sparse embedding with the following process:

1. Perform a dense embedding
2. Apply a linear transformation (to high dimensional space)
3. Make the output k-sparse

The parameters of the dense embedding and the linear transformation are 
learned. Since the fprop may be slow, we keep a candidate set for each word
which consists of the most likely indices to be turned on after the k-max
operation. We record the number of activations for each member of this set,
and periodically resize it to keep only the most active indices. 
Thus the initial training with large candidate sets will be slow, but will
get faster and faster as we restrict their sizes.
]]
local SparseKmax, parent = torch.class('nn.SparseKmax','nn.Module')

--[[
Parameters:
* `vocabSize` - number of entries in the dense lookup table
* `nDenseDim` - number of dimensions for initial dense embedding
* `nSparseDim` - number of dimensions for final sparse embedding
* `k` - number of nonzeros in sparse space
* `nCandidates` - initial size of the candidate set
]]
function SparseKmax:__init(vocabSize,nDenseDim,nSparseDim,k,nCandidates)
  self.nDenseDim = nDenseDim
  self.nSparseDim = nSparseDim
  self.K = k
  self.nCandidates = nCandidates
  self.weights = torch.FloatTensor(nSparseDim,nDenseDim)
  self.counts = torch.FloatTensor(vocabSize,nCandidates)
  self.candidates = torch.ShortTensor(vocabSize,nCandidates)
  self.denseEmbedding = nn.WeightedLookupTable(vocabSize,nDenseDim)
  for i = 1,vocabSize do
    self.candidates[i]:copy(torch.range(1,nCandidates))
  end
  -- Intermediate gradients wrt outputs of dense embeddings.
  self.gradEmbeddings = torch.FloatTensor(1,nDenseDim)
  -- Intermediate gradients wrt inputs to k-max layer.
  self.gradInputKmax = torch.FloatTensor(1,self.K,2)
  -- This stores activations before k-max operation.
  self.activations = torch.FloatTensor(nCandidates,2)
  self.output = torch.FloatTensor(1,self.K,2)
  self:reset()
end

function SparseKmax:reset(stdv)
  self.denseEmbedding:reset(stdv)
  if stdv then
    stdv = stdv * math.sqrt(3)
  else
    stdv = 1./math.sqrt(self.weights:size(2))
  end
  self.counts:zero()
  self.weights:uniform(-stdv,stdv)
end

function SparseKmax:updateOutput(input)
  local nInputs = input:size(1)
  -- Compute outputs of the dense embedding.
  self.dense = self.denseEmbedding:forward(input)
  self.output:resize(nInputs,self.K,2)
  -- Loop over the dense embeddings of the input words.
  for i = 1,input:size(1) do 
    local candidates = self.candidates[input[i][1]]
    -- Copy the indices of candidates into the output.
    self.activations[{{},1}]:copy(candidates)
    self.activations[{{},2}]:zero()
    -- Compute the activations for each element of the candidate set.
    sparse.SaddMtimesDoverlap(self.weights,self.dense[i],self.activations)
    -- Pick top k and copy the scores/indices into the output. 
    -- We sort the indices since this will likely be needed later.
    local topk_val,topk_indx = torch.topk(self.activations[{{},2}],self.K)
    local sorted_indices,sorting_indx 
      = torch.sort(candidates:index(1,topk_indx:long()))
    self.output[{i,{},1}]:copy(sorted_indices)
    self.output[{i,{},2}]:copy(topk_val:index(1,sorting_indx))
    -- Increment counts. 
    for j = 1,self.K do 
      self.counts[input[i][1]][topk_indx[j]] = self.counts[input[i][1]][topk_indx[j]] + 1
    end
  end
  return self.output
end

-- Update the candidate set based on the counts of activations. 
-- `nCandidates` is the size of the new candidate sets.
function SparseKmax:updateCandidateSets(nCandidates)
  self.nCandidates = nCandidates
  local nEntities = self.candidates:size(1)
  local newCandidates = torch.FloatTensor(nEntities,nCandidates)
  -- For each entity, find indices of top activations and keep them.
  for i = 1,nEntities do
    local _,topk = torch.topk(self.counts[i],nCandidates)
    newCandidates[i]:copy(self.candidates[i]:index(1,topk:long()))
  end
  self.candidates = newCandidates
  self.counts:zero()
  self.activations = torch.FloatTensor(nCandidates,2)
end


-- Note, we assume `gradOutput` is sparse since the output is sparse as well.
function SparseKmax:accUpdateGradParameters(input, gradOutput, lr)
  lr = lr or 1
  local nInputs = input:size(1)
  self.gradEmbeddings:resize(nInputs,self.nDenseDim)
  self.gradEmbeddings:zero()
  self.gradInputKmax:resize(nInputs,self.K,2)

  for i = 1,nInputs do
    -- Compute gradients wrt kmax input.
    self.gradInputKmax[{i,{},1}]:copy(self.output[i][{{},1}])
    self.gradInputKmax[{i,{},2}]:zero()
    sparse.SaddSoverlap(self.gradInputKmax[i],gradOutput[i])
    -- Compute gradients wrt dense embedding output.
    sparse.addMtimesS(self.weights:t(),self.gradInputKmax[i],self.gradEmbeddings[i])
  end
  -- Update the weights.
  for i = 1,nInputs do
    sparse.addDouterS(self.denseEmbedding.output[i], self.gradInputKmax[i], self.weights:t(),-lr)
  end
  -- Update the dense embedding weights.
  self.denseEmbedding:accUpdateGradParameters(input,self.gradEmbeddings,lr)
end


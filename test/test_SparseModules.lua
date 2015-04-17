-- Copyright 2004-present Facebook. All Rights Reserved.
-- Compare the sparse modules to the dense ones.
require('fb.luaunit')
local torch = require('fbtorch')

require('nn')
require('fbcunn')
local sparse = require('sparse')
torch.setdefaulttensortype('torch.FloatTensor')
local precision = 1e-5

-- returns a random sparse vector of dimension dim with n nonzero elements
-- both the dense and sparse representations are returned.
function make_sparse_vector(dim,n)
  local s = torch.zeros(dim)
  local indx = torch.randperm(dim)
  for i=1,n do
    s[indx[i]] = torch.randn(1)[1]
  end
  local st = {}
  for i=1,dim do
    if s[i]~=0 then
      table.insert(st,{i,s[i]})
    end
  end
  local sp = torch.Tensor(st)
  return s,sp
end

-- gives dense representation of sparse vector
function to_dense(s,dim)
  local d = torch.zeros(dim)
  for i=1,s:size(1) do
    d[s[i][1]] = s[i][2]
  end
  return d
end

function testSparseConverter()
  local dim = math.random(10,1000)
  local nonzeros = math.random(1,dim/2)
  local d,s = make_sparse_vector(dim,nonzeros)
  -- Test dense to sparse in fprop.
  local model = nn.SparseConverter('DtoS','StoD',dim)
  local y = model:forward(d)
  local err = torch.max(torch.abs(y-s))
  assert(err < precision, 'error in SparseConverter')
  local x = model:updateGradInput(d,y)
  err = torch.max(torch.abs(x-d))
  assert(err < precision, 'error in SparseConverter')
  -- Test sparse to dense.
  local model = nn.SparseConverter('StoD','DtoS',dim)
  local y = model:forward(s)
  local err = torch.max(torch.abs(y-d))
  assert(err < precision, 'error in SparseConverter')
  local x = model:updateGradInput(s,d)
  err = torch.max(torch.abs(x-s))
  assert(err < precision, 'error in SparseConverter')
end

function testSparseSum()
  local dim = math.random(10,1000)
  local nSamples = math.random(5,10)
  local nonzeros = math.random(1,dim/2)
  local denseInput = torch.Tensor(nSamples,dim)
  local sparseInput = torch.Tensor(nSamples,nonzeros,2)

  for i = 1,nSamples do
    local d,s = make_sparse_vector(dim,nonzeros)
    denseInput[i]:copy(d)
    sparseInput[i]:copy(s)
  end

  local denseModel = nn.Sum(1)
  local sparseModel = nn.SparseSum()
  -- Test fprop
  local sparseOutput = sparseModel:forward(sparseInput)
  local denseOutput = denseModel:forward(denseInput)
  local err = torch.max(torch.abs(denseOutput-to_dense(sparseOutput,dim)))
  assert(err < precision,'error in fprop for SparseSum')
  -- Test updateGradInput
  local sparseGradInput = sparseModel:updateGradInput(sparseInput,sparseOutput)
  local denseGradInput = denseModel:updateGradInput(denseInput,denseOutput)
  for i = 1,nSamples do
    local err =
      torch.max(torch.abs(denseGradInput[i]-to_dense(sparseGradInput[i],dim)))
    assert(err < precision,'error in updateGradInput for SparseSum')
  end
end

function testSparseThreshold()
  local dim = math.random(10,1000)
  local nSamples = math.random(5,10)
  local nonzeros = math.random(1,dim/2)
  local denseInput = torch.Tensor(nSamples,dim)
  local sparseInput = torch.Tensor(nSamples,nonzeros,2)

  for i = 1,nSamples do
    local d,s = make_sparse_vector(dim,nonzeros)
    denseInput[i]:copy(d)
    sparseInput[i]:copy(s)
  end

  local denseModel = nn.Threshold()
  local sparseModel = nn.SparseThreshold()
  -- Test fprop.
  local sparseOutput = sparseModel:forward(sparseInput)
  local denseOutput = denseModel:forward(denseInput)
  for i = 1,nSamples do
    local err =
      torch.max(torch.abs(denseOutput[i]-to_dense(sparseOutput[i],dim)))
    assert(err < precision,'error in fprop for SparseThreshold')
  end
  -- Test updateGradInput.
  local sparseGradInput = sparseModel:updateGradInput(sparseInput,sparseOutput)
  local denseGradInput = denseModel:updateGradInput(denseInput,denseOutput)
  for i = 1,nSamples do
    local err =
      torch.max(torch.abs(denseGradInput[i]-to_dense(sparseGradInput[i],dim)))
    assert(err < precision,'error in updateGradInput for SparseThreshold')
  end
end

-- Test a combination of lookup table, sum and threshold.
function testSparseModules()
  local dim = math.random(10,1000)
  local num_entities = math.random(50,200)
  -- Choose a full set of indices, so we can compare to the dense lookup table.
  local num_indices = dim
  -- Create a dense and sparse representation of the same set of weights.
  local weights_dense = torch.Tensor(num_entities,dim)
  local weights_sparse = torch.Tensor(num_entities,num_indices,2)
  for i = 1,num_entities do
    local d,s = make_sparse_vector(dim,num_indices)
    weights_dense[i]:copy(d)
    weights_sparse[i]:copy(s)
  end
  -- Create a dense and sparse version of the same model.
  local denseModel = nn.Sequential()
  denseModel:add(nn.LookupTable(num_entities,dim))
  denseModel:add(nn.Sum(1))
  denseModel:add(nn.Threshold())

  local sparseModel = nn.Sequential()
  sparseModel:add(nn.SparseLookupTable(weights_sparse[{{},{},1}]))
  sparseModel:add(nn.SparseSum())
  sparseModel:add(nn.SparseThreshold())
  -- copy the weights in each
  denseModel:get(1).weight:copy(weights_dense)
  sparseModel:get(1).weight:copy(weights_sparse)

  local num_words = math.random(5,20)
  local words = torch.randperm(num_entities)[{{1,num_words}}]
  -- Input for dense model.
  local denseInput = words
  -- Input for sparse model (must have weights for SparseLookupTable).
  local sparseInput = torch.Tensor(num_words,2):fill(1)
  sparseInput[{{},1}]:copy(words)
  -- Compute outputs.
  local denseOutput = denseModel:forward(denseInput)
  local sparseOutput = sparseModel:forward(sparseInput)
  local err = torch.max(torch.abs(denseOutput-to_dense(sparseOutput,dim)))
  assert(err < precision, 'error in updateOutput')

  -- Make some gradients.
  -- Note that these must have the same indices as the outputs.
  local denseGrad = denseOutput:clone()
  local sparseGrad = sparseOutput:clone()
  denseGrad:normal()
  sparseGrad[{{},2}]:copy(denseGrad)
  -- Test backprop.
  local lr = math.abs(torch.randn(1)[1])
  denseModel:updateGradInput(denseInput,denseGrad)
  sparseModel:updateGradInput(sparseInput,sparseGrad)
  denseModel:accUpdateGradParameters(denseInput,denseGrad,lr)
  sparseModel:accUpdateGradParameters(sparseInput,sparseGrad,lr)
  -- Make sure the updated weights are the same.
  local denseWeights = denseModel:get(1).weight
  local sparseWeights = sparseModel:get(1).weight
  for i = 1,num_entities do
    local err =
      torch.max(torch.abs(denseWeights[i]-to_dense(sparseWeights[i],dim)))
    assert(err < precision, 'error in accUpdateGradParameters')
  end
end

-- Sanity check: make sure it's equivalent to lookup+linear with full outputs
function testSparseKmax()
  local num_entities = math.random(20,50)
  local inputSize = math.random(10,20)
  local outputSize = math.random(30,50)
  local k = outputSize
  local nCandidates = outputSize

  -- Function to check a dense model and a sparse model are the same.
  function compareDenseToSparse(sparseModel,denseModel)
    local num_words = math.random(2,15)--num_entities)
    local words_indx = torch.randperm(num_entities)[{{1,num_words}}]
    local words = torch.FloatTensor(num_words,2)
    words[{{},1}]:copy(words_indx)
    words[{{},2}]:fill(1)
    -- Test fprop.
    local denseOutput = denseModel:forward(words)
    local sparseOutput = sparseModel:forward(words)
    for i = 1,num_words do
      local err = torch.max(torch.abs(torch.Tensor(denseOutput[i])
                            -to_dense(sparseOutput[i],outputSize)))
      assert(err < precision, 'error in fprop of SparseKmax')
    end
    -- Test backprop.
    local learningRate = math.abs(torch.randn(1)[1])
    denseGrad = torch.Tensor(num_words,outputSize)
    sparseGrad = torch.Tensor(num_words,outputSize,2)
    for i = 1,num_words do
      local d,s = make_sparse_vector(outputSize,outputSize)
      denseGrad[i]:copy(d)
      sparseGrad[i]:copy(s)
    end
    denseModel:updateGradInput(words,denseGrad)
    denseModel:accUpdateGradParameters(words,denseGrad,learningRate)
    sparseModel:accUpdateGradParameters(words,sparseGrad,learningRate)
    -- Compare gradients.
    local gs = sparseModel.gradEmbeddings
    local gd = denseModel:get(2).gradInput
    err = torch.max(torch.abs(gs-gd))
    assert(err < precision, 'error in gradients')
    -- Compare linear layer weights.
    local ws = sparseModel.weights
    local wd = denseModel:get(2).weight
    err = torch.max(torch.abs(ws-wd))
    assert(err < precision, 'error in linear weights')
    -- Compare embedding weights.
    local es = sparseModel.denseEmbedding.weight
    local ed = denseModel:get(1).weight
    err = torch.max(torch.abs(es-ed))
    assert(err < precision, 'error in embedding weights')
    -- At least make sure resizing candidate sets doesn't crash...
    sparseModel:updateCandidateSets(nCandidates/2)
    sparseModel.K = nCandidates/2
    sparseModel:forward(words)
    sparseModel:accUpdateGradParameters(words,sparseGrad,learningRate)
  end

  -- Make a dense model and an equivalent sparse one.
  local denseModel = nn.Sequential()
  denseModel:add(nn.WeightedLookupTable(num_entities,inputSize))
  denseModel:add(nn.Linear(inputSize,outputSize))
  denseModel:get(2).bias:zero()

  local sparseModel =
    nn.SparseKmax(num_entities,inputSize,outputSize,k,nCandidates)
  sparseModel.denseEmbedding.weight:copy(denseModel:get(1).weight)
  sparseModel.weights:copy(denseModel:get(2).weight)

  -- Compare them.
  compareDenseToSparse(sparseModel,denseModel)
end

-- Similar to above, but with a SparseSum module and a SparseConverter
-- on the gradients.
function testSparseKmax2()
  local num_entities = math.random(20,50)
  local inputSize = math.random(10,20)
  local outputSize = math.random(30,50)
  local k = outputSize
  local nCandidates = outputSize

  -- Function to check a dense model and a sparse model are the same.
  function compareDenseToSparse(sparseModel,denseModel)
    local num_words = math.random(2,15)--num_entities)
    local words_indx = torch.randperm(num_entities)[{{1,num_words}}]
    local words = torch.FloatTensor(num_words,2)
    words[{{},1}]:copy(words_indx)
    words[{{},2}]:fill(1)
    -- Test fprop.
    local denseOutput = denseModel:forward(words)
    local sparseOutput = sparseModel:forward(words)
    local err = torch.max(torch.abs(denseOutput-to_dense(sparseOutput,outputSize)))
    assert(err < precision, 'error in fprop of SparseKmax')
    -- Test backprop.
    local learningRate = math.abs(torch.randn(1)[1])
    local denseGrad,_ = make_sparse_vector(outputSize,outputSize)
    sparseGrad = denseGrad
    local nw1 = sparseModel:get(1).weights:norm()
    local ne1 = sparseModel:get(1).denseEmbedding.weight:norm()
    denseModel:updateGradInput(words,denseGrad)
    denseModel:accUpdateGradParameters(words,denseGrad,learningRate)
    sparseModel:updateGradInput(words,sparseGrad)
    sparseModel:accUpdateGradParameters(words,sparseGrad,learningRate)
    local nw2 = sparseModel:get(1).weights:norm()
    local ne2 = sparseModel:get(1).denseEmbedding.weight:norm()
    assert(math.abs(nw1-nw2) > precision, 'weights did not change')
    assert(math.abs(ne1-ne2) > precision, 'embedding weights did not change')
    -- Compare gradients.
    local gs = sparseModel:get(1).gradEmbeddings
    local gd = denseModel:get(2).gradInput
    err = torch.max(torch.abs(gs-gd))
    assert(err < precision, 'error in gradients')
    -- Compare linear layer weights.
    local ws = sparseModel:get(1).weights
    local wd = denseModel:get(2).weight
    err = torch.max(torch.abs(ws-wd))
    assert(err < precision, 'error in linear weights')
    -- Compare embedding weights.
    local es = sparseModel:get(1).denseEmbedding.weight
    local ed = denseModel:get(1).weight
    err = torch.max(torch.abs(es-ed))
    assert(err < precision, 'error in embedding weights')
    -- At least make sure resizing candidate sets doesn't crash...
    sparseModel:get(1):updateCandidateSets(nCandidates/2)
    sparseModel:get(1).K = nCandidates/2
    sparseModel:forward(words)
    sparseModel:accUpdateGradParameters(words,sparseGrad,learningRate)
  end

  -- Make a dense model and an equivalent sparse one.
  local denseModel = nn.Sequential()
  denseModel:add(nn.WeightedLookupTable(num_entities,inputSize))
  denseModel:add(nn.Linear(inputSize,outputSize))
  denseModel:add(nn.Sum(1))
  denseModel:get(2).bias:zero()

  local sparseModel = nn.Sequential()
  sparseModel:add(nn.SparseKmax(num_entities,inputSize,outputSize,k,nCandidates))
  sparseModel:add(nn.SparseSum())
  sparseModel:add(nn.SparseConverter(nil,'DtoS'))
  sparseModel:get(1).denseEmbedding.weight:copy(denseModel:get(1).weight)
  sparseModel:get(1).weights:copy(denseModel:get(2).weight)

  -- Compare them.
  compareDenseToSparse(sparseModel,denseModel)
end

-- Run a bunch of times.
function testAllInLoop()
    for i = 1,10 do
        testSparseSum()
        testSparseThreshold()
        testSparseModules()
        testSparseKmax()
        testSparseKmax2()
    end
end

LuaUnit:main()

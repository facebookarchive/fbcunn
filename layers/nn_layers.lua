-- Copyright 2004-present Facebook. All Rights Reserved.

require('nn')

local mod_path = (...):match('(.+)[.][%w_]+$')

-- List all layers here
require(mod_path .. '.CrossMapNormalization')
require(mod_path .. '.LocallyConnected')
require(mod_path .. '.ClassHierarchicalNLLCriterion')
require(mod_path .. '.HSM')
require(mod_path .. '.SparseNLLCriterion')
require(mod_path .. '.SparseThreshold')
require(mod_path .. '.SparseSum')
require(mod_path .. '.SparseLookupTable')
require(mod_path .. '.SparseKmax')
require(mod_path .. '.SparseConverter')
require(mod_path .. '.KMaxPooling')
require(mod_path .. '.GroupKMaxPooling')
require(mod_path .. '.WeightedLookupTable')
require(mod_path .. '.LinearNB')

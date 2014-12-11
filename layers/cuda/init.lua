-- Copyright 2004-present Facebook. All Rights Reserved.

-- Note that the module name is fbcunn.layers.cuda,
-- not fbcunn.layers.cuda.init.
local mod_path = ...
local parent_path = mod_path:match('(.+)[.][%w_]+$')

require(parent_path .. '.nn_layers')
require(mod_path .. '.HalfPrecision')  -- cuda only
require(mod_path .. '.TemporalConvolutionFB')  -- cuda only
require(mod_path .. '.TemporalKMaxPooling')  -- cuda only
require(mod_path .. '.OneBitQuantization') -- cuda only
require(mod_path .. '.FeatureLPPooling') -- cuda only

require(parent_path .. '.cuda_ext')

-- Copyright 2004-present Facebook. All Rights Reserved.

require('cutorch')
require('nn')

local cur_path = ...
local mod_path = (...):match('(.+)[.][%w_]+$')

-- Require both freaking local lua_extension and lua source file
-- otherwise the function into which LUAOPEN expands
-- will NOT be called.
require(mod_path .. '.cufft_lua_ext')
require(mod_path .. '.SpatialConvolutionCuFFT')  -- cuda only

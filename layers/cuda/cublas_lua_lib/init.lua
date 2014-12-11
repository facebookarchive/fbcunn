-- Copyright 2004-present Facebook. All Rights Reserved.

require('nn')

local mod_path = (...):match('(.+)[.][%w_]+$')

-- Require both freaking local lua_extension and lua source file
-- otherwise the function into which LUAOPEN expands
-- will NOT be called.
require(mod_path .. '.cublas_lua_ext')
require(mod_path .. '.CuBLASWrapper')  -- cuda only

-- Copyright 2004-present Facebook. All Rights Reserved.

-- Note that the module name is fbcunn.layers.cpu,
-- not fbcunn.layers.cpu.init.
local mod_path = ...
local parent_path = mod_path:match('(.+)[.][%w_]+$')

require(parent_path .. '.nn_layers')
require(parent_path .. '.cpu_ext')

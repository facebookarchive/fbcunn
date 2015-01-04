--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
local ffi = require 'ffi'
local Threads = require 'threads'

-- This script contains the logic to create K threads for parallel data-loading.
-- For the data-loading details, look at donkey.lua
-------------------------------------------------------------------------------
do -- start K datathreads (donkeys)
   local options = opt -- make an upvalue to serialize over to donkey threads
   donkeys = Threads(
      opt.nDonkeys,
      function()
         gsdl = require 'sdl2'
         require 'torch'
      end,
      function(idx)
         opt = options -- pass to all donkeys via upvalue
         tid = idx
         local seed = opt.manualSeed + idx
         torch.manualSeed(seed)
         print(string.format('Starting donkey with id: %d seed: %d', tid, seed))
         paths.dofile('donkey.lua')
      end
   );
end

nClasses = nil
classes = nil
donkeys:addjob(function() return trainLoader.classes end, function(c) classes = c end)
donkeys:synchronize()
nClasses = #classes
assert(nClasses, "Failed to get nClasses")
print('nClasses: ', nClasses)
torch.save(paths.concat(opt.save, 'classes.t7'), classes)

nTest = 0
donkeys:addjob(function() return testLoader:sizeTest() end, function(c) nTest = c end)
donkeys:synchronize()
assert(nTest > 0, "Failed to get nTest")
print('nTest: ', nTest)


------ Some FFI stuff used to pass storages between threads ------------------
ffi.cdef[[
void THFloatStorage_free(THFloatStorage *self);
void THLongStorage_free(THLongStorage *self);
]]

function setFloatStorage(tensor, storage_p)
   assert(storage_p and storage_p ~= 0, "FloatStorage is NULL pointer");
   local cstorage = ffi.cast('THFloatStorage*', torch.pointer(tensor:storage()))
   if cstorage ~= nil then
      ffi.C['THFloatStorage_free'](cstorage)
   end
   local storage = ffi.cast('THFloatStorage*', storage_p)
   tensor:cdata().storage = storage
end

function setLongStorage(tensor, storage_p)
   assert(storage_p and storage_p ~= 0, "LongStorage is NULL pointer");
   local cstorage = ffi.cast('THLongStorage*', torch.pointer(tensor:storage()))
   if cstorage ~= nil then
      ffi.C['THLongStorage_free'](cstorage)
   end
   local storage = ffi.cast('THLongStorage*', storage_p)
   tensor:cdata().storage = storage
end

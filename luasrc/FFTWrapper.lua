-- Copyright 2004-present Facebook. All Rights Reserved.

local FFTWrapper = torch.class('nn.FFTWrapper')

function FFTWrapper:__init(cufft)
   self.batchDims = 0
   self.cufft = cufft or 1
end

function FFTWrapper:fft(time, frequency, batchDims)
   assert(batchDims >= 1)
   assert(batchDims <= 2)
   self.batchDims = batchDims
   time.nn.FFTWrapper_fft(self, time, frequency)
end

function FFTWrapper:ffti(time, frequency, batchDims)
   assert(batchDims >= 1)
   assert(batchDims <= 2)
   self.batchDims = batchDims
   time.nn.FFTWrapper_ffti(self, time, frequency)
end

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
    -- If calling fft from lua directly, just pass a buffer in any case.
    -- In practice it is only really needed for 2d-fft of size > 32
    local buffer = {}
    if self.cufft == 1 then
        if #frequency:size() == 4 then
            assert(frequency:size()[2] / 2 + 1 == frequency:size()[3])
        end
        -- Need to allocate explicit cufft plans, a buffer is not enough
        buffer = torch.CudaTensor(torch.LongStorage({1, 1, 1, 1}))
    else
        if #frequency:size() == 4 then
            assert(frequency:size()[3] / 2 + 1 == frequency:size()[2])
        end
        buffer = frequency:clone()
    end
    time.nn.FFTWrapper_fft(self, time, frequency, buffer)
end

function FFTWrapper:ffti(time, frequency, batchDims)
    assert(batchDims >= 1)
    assert(batchDims <= 2)
    self.batchDims = batchDims
    -- In practice it is only really needed for 2d-fft of size > 32
    local size = frequency:size()
    local bufferSize = {}
    local buffer = torch.CudaTensor(torch.LongStorage({1, 1, 1, 1}))
    -- Make full buffer to hold the whole complex tensor if needed
    if self.cufft == 1 then
        if #time:size() - batchDims == 2 then
            assert(size[2] / 2 + 1 == size[3])
        end
    elseif batchDims == 1 and #size == 4 then
        if batchDims == 1 and #size == 4 then
            assert(size[3] / 2 + 1 == size[2])
            --
            bufferSize = torch.LongStorage({size[1], size[3], size[3], size[4]})
            buffer = torch.CudaTensor(bufferSize)
        else
            buffer = frequency:clone()
        end
    end
    time.nn.FFTWrapper_ffti(self, time, frequency, buffer)
end

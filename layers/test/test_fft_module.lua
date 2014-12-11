require 'libFFTconv'
require 'cunn'
require 'math'

require 'fbcode.deeplearning.torch.layers.nyu.cuda'
require 'fbcunn.layers.nn_layers'
require 'fbcunn.layers.cuda.fft.fft_wrapper_lua_lib'

local mytester = torch.Tester()

local test = {}
local printResults = false

-- We exploit hermitian symmetry to write out only 1/2 the data.
-- CuFFT exploits hermitian symmetry along the innermost dimension
-- FBFFT is parameteriazble  only determined by the output tensor dimesions.
-- Ideally we would use outermost dimension hermitian symmetry for better
-- coalescing but if we check correctness vs CuFFT then we match it.
local runTests = true
local matchCuFFTAlloc = runTests
local nTests = 1
if runTests then
    nTests = 1
end

-- Type of initialization:
-- 1: fill(1.0f)
-- 2: 1.0f if 0 mod 2 else 2.0f
-- 3: starts at 6.0f and += 1.0f at each entry
-- else: random
local init = 4

-- {FFTDim, {numBatches, FFTSizes}}
local problemSizes = {
  {1, {128, 2}},
  {1, {128, 4}},
  {1, {128, 8}},
  {1, {128, 16}},
  {1, {128, 32}},
  {1, {128, 64}},
  {1, {128, 128}},
  {1, {128, 256}},

  {1, {128 * 128, 2}},
  {1, {128 * 128, 4}},
  {1, {128 * 128, 8}},
  {1, {128 * 128, 16}},
  {1, {128 * 128, 32}},
  {1, {128 * 128, 64}},
  {1, {128 * 128, 128}},
  {1, {128 * 128, 256}},

  {2, {4, 2, 2}},
  {2, {4, 4 ,4}},
  {2, {4, 8, 8}},
  {2, {4, 16, 16}},
  {2, {4, 32, 32}},

  {2, {128, 2, 2}},
  {2, {128, 4 ,4}},
  {2, {128, 8, 8}},
  {2, {128, 16, 16}},
  {2, {128, 32, 32}},

  {2, {128 * 128, 2, 2}},
  {2, {128 * 128, 4 ,4}},
  {2, {128 * 128, 8, 8}},
  {2, {128 * 128, 16, 16}},
  {2, {128 * 128, 32, 32}},

}

local function concat(t1,t2)
    local res = {}
    for i=1,#t1 do
        res[#res + 1] = t1[i]
    end
    for i=1,#t2 do
        res[#res + 1] = t2[i]
    end
    return res
end

local function benchmarkNYUFFT(problemSize, time)
    if not (problemSize[1] == 2) then
        return nil;
    end
    if problemSize[2][2] > 64 or problemSize[2][3] > 64 then
        return nil
    end

    local a = nn.Sequential()
    a:add(nn.FFT())
    -- a:add(nn.FFTI(problemSize[2][2], problemSize[2][3]))

    local frequency = a:forward(time)
    if printResults then
        print(frequency)
    end
end

local function benchmarkCuFFT(problemSize, time)
    local fftDim   = problemSize[1]
    local timeSize = problemSize[2]
    local freqSize = {}
    for i = 1, #timeSize do
      if i == #timeSize then
        freqSize = concat(freqSize, {math.floor(timeSize[i] / 2) + 1})
      else
        freqSize = concat(freqSize, {timeSize[i]})
      end
    end
    freqSize = concat(freqSize, {2})

    local frequency = torch.CudaTensor(torch.LongStorage(freqSize))

    local batchDims = #timeSize - fftDim
    local net = nn.FFTWrapper()
    net:fft(time, frequency, batchDims)
--    net:ffti(time, frequency, batchDims)

    if printResults then
        print(frequency)
    end

    return frequency
end

local function benchmarkFBFFT(problemSize, time, CuFFTAlloc)
    local fftDim   = problemSize[1]
    local timeSize = problemSize[2]
    local freqSize = {}
    local hermitianDim = -1
    if CuFFTAlloc then
        hermitianDim = #timeSize
    else
        hermitianDim = #timeSize - fftDim + 1
    end
    for i = 1, #timeSize do
        if i == hermitianDim then
            freqSize = concat(freqSize, {math.floor(timeSize[i] / 2) + 1})
        else
            freqSize = concat(freqSize, {timeSize[i]})
        end
    end
    freqSize = concat(freqSize, {2})

    local frequency = torch.CudaTensor(torch.LongStorage(freqSize))

    local batchDims = #timeSize - fftDim
    local cufft = 0
    local net = nn.FFTWrapper(cufft)
    net:fft(time, frequency, batchDims)
--    net:ffti(time, frequency, batchDims)

    if printResults then
        print(frequency)
    end

    return frequency
end

function test.test()
    for t = 1, nTests do
        for i = 1, #problemSizes do
          local timeTensor = {}

          -- Init with proper entries
          if init == 1 then
            timeTensor = torch.CudaTensor(
                torch.LongStorage(problemSizes[i][2])):fill(1.0)
          elseif init == 2 then
              local val = 1
              timeTensor = torch.Tensor(torch.LongStorage(
                  problemSizes[i][2])):apply(function()
                      val = val + 1
                      if val % 2 == 0 then
                          return 1
                      else
                          return 2
                      end
                  end):cuda()
          elseif init == 3 then
              local val = 5
              timeTensor = torch.Tensor(
                  torch.LongStorage(problemSizes[i][2])):apply(function()
                      val = val + 1
                      return val
                  end):cuda()
          else
              timeTensor = torch.CudaTensor(
                  torch.LongStorage(problemSizes[i][2])):normal()
          end

          if printResults then
              print(timeTensor)
          end

          local function assertdiff(cufft, fbfft)
              local m = (cufft:double() - fbfft:double()):abs():max()
              local n = cufft:norm()
              if printResults then
                  print(m, n, m / n)
              end
              assert(m / n < 1e-3)
              if m / n > 1e-3 then
                  print('DOES NOT PAS CHECK !!')
                  print(m, n, m / n)
              end
          end

          local fbfft =
              benchmarkFBFFT(problemSizes[i], timeTensor, matchCuFFTAlloc)
          if runTests then
              local cufft  = benchmarkCuFFT(problemSizes[i], timeTensor)
              benchmarkNYUFFT(problemSizes[i], timeTensor)
              -- check only against cufft, nyufft has sizing issues
              assertdiff(cufft, fbfft)
          end

          collectgarbage()

        end
    end
end

mytester:add(test)
mytester:run()

require 'cunn'
require 'math'

require 'fbcunn'

require('fb.luaunit')
require('fbtorch')

local mytester = torch.Tester()

local FFTTester = {}
local printResults = false
local precision = 2e-7

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

-- {FFTDim, {numBatches, FFTSizes}}
local _iclr2015TestCases = {
  {1, {4, 8}},
  {1, {4, 16}},
  {1, {4, 32}},
  {1, {4, 64}},
  {1, {4, 128}},
  {1, {4, 256}},

  {1, {32, 8}},
  {1, {32, 16}},
  {1, {32, 32}},
  {1, {32, 64}},
  {1, {32, 128}},
  {1, {32, 256}},

  {1, {128, 8}},
  {1, {128, 16}},
  {1, {128, 32}},
  {1, {128, 64}},
  {1, {128, 128}},
  {1, {128, 256}},

  {1, {1024, 8}},
  {1, {1024, 16}},
  {1, {1024, 32}},
  {1, {1024, 64}},
  {1, {1024, 128}},
  {1, {1024, 256}},

  {1, {4096, 8}},
  {1, {4096, 16}},
  {1, {4096, 32}},
  {1, {4096, 64}},
  {1, {4096, 128}},
  {1, {4096, 256}},

  {1, {128 * 128, 8}},
  {1, {128 * 128, 16}},
  {1, {128 * 128, 32}},
  {1, {128 * 128, 64}},
  {1, {128 * 128, 128}},
  {1, {128 * 128, 256}},

  {1, {256 * 256, 8}},
  {1, {256 * 256, 16}},
  {1, {256 * 256, 32}},
  {1, {256 * 256, 64}},
  {1, {256 * 256, 128}},
  {1, {256 * 256, 256}},

  {2, {4, 8, 8}},
  {2, {4, 16, 16}},
  {2, {4, 32, 32}},
  {2, {4, 64, 64}},
  {2, {4, 128, 128}},

  {2, {32, 8, 8}},
  {2, {32, 16, 16}},
  {2, {32, 32, 32}},
  {2, {32, 64, 64}},
  {2, {32, 128, 128}},

  {2, {128, 8, 8}},
  {2, {128, 16, 16}},
  {2, {128, 32, 32}},
  {2, {128, 64, 64}},
  {2, {128, 128, 128}},

  {2, {1024, 8, 8}},
  {2, {1024, 16, 16}},
  {2, {1024, 32, 32}},
  {2, {1024, 64, 64}},
  {2, {1024, 128, 128}},

  {2, {4096, 8, 8}},
  {2, {4096, 16, 16}},
  {2, {4096, 32, 32}},
  {2, {4096, 64, 64}},
  {2, {4096, 128, 128}},

  {2, {128 * 128, 8, 8}},
  {2, {128 * 128, 16, 16}},
  {2, {128 * 128, 32, 32}},
  {2, {128 * 128, 64, 64}},
  {2, {128 * 128, 128, 128}},

--[[
  {2, {256 * 256, 8, 8}},
  {2, {256 * 256, 16, 16}},
  {2, {256 * 256, 32, 32}},
  {2, {256 * 256, 64, 64}},
  {2, {256 * 256, 128, 128}},
--]]
}

local _stressTestCases = {
  {1, {32 * 32, 2}},
  {1, {32 * 32, 4}},
  {1, {32 * 32, 8}},
  {1, {32 * 32, 16}},
  {1, {32 * 32, 32}},
  {1, {32 * 32, 64}},
  {1, {32 * 32, 128}},
  {1, {32 * 32, 256}},

  {2, {32 * 32, 2, 2}},
  {2, {32 * 32, 4, 4}},
  {2, {32 * 32, 8, 8}},
  {2, {32 * 32, 16, 16}},
  {2, {32 * 32, 32, 32}},
  {2, {32 * 32, 64, 64}},
  {2, {32 * 32, 128, 128}},

  {1, {64 * 64, 2}},
  {1, {64 * 64, 4}},
  {1, {64 * 64, 8}},
  {1, {64 * 64, 16}},
  {1, {64 * 64, 32}},
  {1, {64 * 64, 64}},
  {1, {64 * 64, 128}},
  {1, {64 * 64, 256}},

  {2, {64 * 64, 2, 2}},
  {2, {64 * 64, 4 ,4}},
  {2, {64 * 64, 8, 8}},
  {2, {64 * 64, 16, 16}},
  {2, {64 * 64, 32, 32}},
  {2, {64 * 64, 64, 64}},
  {2, {64 * 64, 128, 128}},

  {1, {128 * 128, 2}},
  {1, {128 * 128, 4}},
  {1, {128 * 128, 8}},
  {1, {128 * 128, 16}},
  {1, {128 * 128, 32}},
  {1, {128 * 128, 64}},
  {1, {128 * 128, 128}},
  {1, {128 * 128, 256}},

  {2, {128 * 128, 2, 2}},
  {2, {128 * 128, 4 ,4}},
  {2, {128 * 128, 8, 8}},
  {2, {128 * 128, 16, 16}},
  {2, {128 * 128, 32, 32}},
  {2, {128 * 128, 64, 64}},
  {2, {128 * 128, 128, 128}},

}

local testCases = {
  {1, {127, 2}},
  {1, {127, 4}},
  {1, {127, 8}},
  {1, {127, 16}},
  {1, {127, 32}},
  {1, {127, 64}},
  {1, {127, 128}},
  {1, {127, 256}},

  {1, {437, 2}},
  {1, {437, 4}},
  {1, {437, 8}},
  {1, {437, 16}},
  {1, {437, 32}},
  {1, {437, 64}},
  {1, {437, 128}},
  {1, {437, 256}},

  {2, {9, 2, 2}},
  {2, {9, 4 ,4}},
  {2, {9, 8, 8}},
  {2, {9, 16, 16}},
  {2, {9, 32, 32}},
  {2, {9, 64, 64}},
  {2, {9, 128, 128}},

  {2, {128, 2, 2}},
  {2, {128, 4 ,4}},
  {2, {128, 8, 8}},
  {2, {128, 16, 16}},
  {2, {128, 32, 32}},
  {2, {128, 64, 64}},
  {2, {128, 128, 128}},

  {2, {1, 2, 2}},
  {2, {1, 4 ,4}},
  {2, {1, 8, 8}},
  {2, {4, 16, 16}},
  {2, {4, 32, 32}},
  {2, {4, 64, 64}},
  {2, {4, 128, 128}},

  {2, {1, 64, 64}},
  {2, {4, 64, 64}},
  {2, {8, 64, 64}},
  {2, {16, 64, 64}},
  {2, {32, 64, 64}},
  {2, {64, 64, 64}},
  {2, {128, 64, 64}},

  {2, {1, 128, 128}},
  {2, {4, 128, 128}},
  {2, {8, 128, 128}},
  {2, {16, 128, 128}},
  {2, {32, 128, 128}},

  {2, {1, 16, 16}},
  {2, {4, 16, 16}},
  {2, {8, 16, 16}},
  {2, {16, 16, 16}},
  {2, {32, 16, 16}},
  {2, {64, 16, 16}},
  {2, {128, 16, 16}},
  {2, {256, 16, 16}},
  {2, {512, 16, 16}},

  {2, {1, 32, 32}},
  {2, {4, 32, 32}},
  {2, {8, 32, 32}},
  {2, {16, 32, 32}},
  {2, {32, 32, 32}},
  {2, {64, 32, 32}},
  {2, {128, 32, 32}},
  {2, {256, 32, 32}},
  {2, {512, 32, 32}},
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

local function benchmarkCuFFT(problemSize, timeCuda)
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


    local timeInvCuda = timeCuda:clone()
    local frequencyCuda =
        torch.CudaTensor(torch.LongStorage(freqSize)):fill(-47.0)

    local batchDims = #timeSize - fftDim
    local net = nn.FFTWrapper(1)
    net:fft(timeCuda, frequencyCuda, batchDims)
    net:ffti(timeInvCuda, frequencyCuda, batchDims)

    local timeInv = timeInvCuda:double()
    local frequency = frequencyCuda:double()
    if printResults then
        print('forward re:', frequency:select(fftDim + 2, 1))
        print('forward im:', frequency:select(fftDim + 2, 2))

        print('inverse re:', timeInv)
    end

    timeInvCuda = {}
    frequencyCuda = {}
    collectgarbage()

    return frequency, timeInv
end

local function benchmarkFBFFT(problemSize, timeCuda, frequency2)
    local fftDim   = problemSize[1]
    local timeSize = problemSize[2]
    local freqSize = {}
    local hermitianDim = #timeSize - fftDim + 1

    for i = 1, #timeSize do
        if i == hermitianDim then
            freqSize = concat(freqSize, {math.floor(timeSize[i] / 2) + 1})
        else
            freqSize = concat(freqSize, {timeSize[i]})
        end
    end
    freqSize = concat(freqSize, {2})

    local timeInvCuda = timeCuda:clone()
    local frequencyCuda =
        torch.CudaTensor(torch.LongStorage(freqSize)):fill(-47.0)
    local batchDims = #timeSize - fftDim
    local net = nn.FFTWrapper(0)
    net:fft(timeCuda, frequencyCuda, batchDims)
    net:ffti(timeInvCuda, frequencyCuda, batchDims)

    local timeInv = timeInvCuda:double()
    local frequency = frequencyCuda:double()
    if fftDim == 2 then
        frequency = frequency:transpose(2, 3)
    end

    if printResults then
        print('forward re:', frequency:select(fftDim + 2, 1))
        print('forward im:', frequency:select(fftDim + 2, 2))

        print('inverse re:', timeInv)
    end

    timeInvCuda = {}
    frequencyCuda = {}
    collectgarbage()

    return frequency, timeInv
end

local function initCuda(ps, localInit)
    local timeCudaTensor = {}
    -- LocalInit with proper entries
    if localInit == 1 then
      timeCudaTensor = torch.CudaTensor(
          torch.LongStorage(ps)):fill(1.0)
    elseif localInit == 2 then
        local val = -1
        timeCudaTensor = torch.Tensor(torch.LongStorage(
            ps)):apply(function()
          val = val + 1
          return val % 2 + 1
            end):cuda()
    elseif localInit == 3 then
        local val = -1
        timeCudaTensor = torch.Tensor(torch.LongStorage(
            ps)):apply(function()
          val = val + 1
          return val % 4 + 1
            end):cuda()
    elseif localInit == 4 then
        local val = 0
        local res = 0
        timeCudaTensor = torch.Tensor(
            torch.LongStorage(ps)):apply(function()
          val = val + 1
          if val == ps[#ps] + 1 then
              val = 1
              res = res + 1
          end
          return res
            end):cuda()
    elseif localInit == 5 then
        local val = 0
        timeCudaTensor = torch.Tensor(
            torch.LongStorage(ps)):apply(function()
          val = val + 1
          return val
            end):cuda()
    else
        timeCudaTensor = torch.CudaTensor(
            torch.LongStorage(ps)):normal()
    end
    return timeCudaTensor
end

local function run(localInit, problemSizes)
    for t = 1, nTests do
        for i = 1, #problemSizes do
          local timeCudaTensor = {}
          local fftDim = problemSizes[i][1]
          local ps = problemSizes[i][2]

          timeCudaTensor = initCuda(ps, localInit)

          if printResults then
              print(timeCudaTensor:float())
          end

          local function assertdiff(reffft, fbfft, fftDim, fftSize)
              if ps[1] > 512 then
                  print('Skip horrendously long test, need to transpose',
                        ' the data efficiently to test')
                  return
              end
              local m = (reffft:double() - fbfft:double()):abs():max()
              local n = reffft:double():norm() + 1e-10
              local nfbfft = fbfft:double():norm() + 1e-10
              if m / n > precision then
                  if printResults then
                      print('Check max diff, norm, norm fbfft, max normalized = ',
                            m, n, nfbfft, m / n)
                      print('FAILS CHECK !!')
                      print(m, n, m / n)
                  end
              end
              assert(m / n < precision)
              return
          end

          local fbfft, fbifft =
              benchmarkFBFFT(problemSizes[i], timeCudaTensor, matchCuFFTAlloc)
          local cufft, cuifft = benchmarkCuFFT(problemSizes[i], timeCudaTensor)
          if runTests then
              assertdiff(cufft, fbfft, fftDim, ps[2])
              assertdiff(cuifft, fbifft, fftDim, ps[2])
          end

          timeCudaTensor = {}
          collectgarbage()
        end
    end
end

function FFTTester.test()
-- Type of initialization:
-- 1: fill(1.0f)
-- 2: 1.0f if 0 mod 2 else 2.0f
-- 3: val % 4 + 1
-- 4: val == row
-- 5: starts at 1.0f and += 1.0f at each entry
-- else: random
    local localInits = {1, 2, 3, 4, 5, 6}
    for i = 1, #localInits do
        run(localInits[i], testCases)
        collectgarbage()
        cutorch.synchronize()
    end
end

mytester:add(FFTTester)
mytester:run()

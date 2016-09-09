require 'cunn'
require 'math'

require 'fbcunn'

require('fb.luaunit')
require('fbtorch')

local mytester = torch.Tester()

local FFTTester = {}
local printResults = false
local precision = 2 * 2e-7 -- 2 ULPs relative to the largest input

-- We exploit hermitian symmetry to write out only 1/2 the data.
-- CuFFT exploits hermitian symmetry along the innermost dimension
-- FBFFT is parametriazble  only determined by the output tensor dimensions.
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

  {1, {32, 8}},
  {1, {32, 16}},
  {1, {32, 32}},
  {1, {32, 64}},
  {1, {32, 128}},

  {1, {128, 8}},
  {1, {128, 16}},
  {1, {128, 32}},
  {1, {128, 64}},
  {1, {128, 128}},

  {1, {1024, 8}},
  {1, {1024, 16}},
  {1, {1024, 32}},
  {1, {1024, 64}},
  {1, {1024, 128}},

  {1, {4096, 8}},
  {1, {4096, 16}},
  {1, {4096, 32}},
  {1, {4096, 64}},
  {1, {4096, 128}},

  {1, {128 * 128, 8}},
  {1, {128 * 128, 16}},
  {1, {128 * 128, 32}},
  {1, {128 * 128, 64}},
  {1, {128 * 128, 128}},

  {1, {256 * 256, 8}},
  {1, {256 * 256, 16}},
  {1, {256 * 256, 32}},
  {1, {256 * 256, 64}},
  {1, {256 * 256, 128}},

  {2, {4, 8, 8}},
  {2, {4, 16, 16}},
  {2, {4, 32, 32}},
  {2, {4, 64, 64}},

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

  {2, {1, 8, 8}},
  {2, {1, 16, 16}},
  {2, {1, 32, 32}},
  {2, {1, 64, 64}},

  {2, {128 * 128, 8, 8}},
  {2, {128 * 128, 16, 16}},
  {2, {128 * 128, 32, 32}},
  {2, {128 * 128, 64, 64}},
  {2, {128 * 128, 128, 128}},

  {2, {256 * 256, 8, 8}},
  {2, {256 * 256, 16, 16}},
  {2, {256 * 256, 32, 32}},
  {2, {256 * 256, 64, 64}},
-- Too much memory
-- {2, {256 * 256, 128, 128}},
}

local _stressTestCases = {
  {1, {32 * 32, 2}},
  {1, {32 * 32, 4}},
  {1, {32 * 32, 8}},
  {1, {32 * 32, 16}},
  {1, {32 * 32, 32}},
  {1, {32 * 32, 64}},
  {1, {32 * 32, 128}},

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

  {2, {128 * 128, 2, 2}},
  {2, {128 * 128, 4 ,4}},
  {2, {128 * 128, 8, 8}},
  {2, {128 * 128, 16, 16}},
  {2, {128 * 128, 32, 32}},
  {2, {128 * 128, 64, 64}},
  {2, {128 * 128, 128, 128}},

}

local testCases = {
  {1, {1, 2}},
  {1, {1, 4}},
  {1, {1, 8}},
  {1, {1, 16}},
  {1, {1, 32}},
  {1, {1, 64}},
  {1, {1, 128}},

  {1, {127, 2}},
  {1, {127, 4}},
  {1, {127, 8}},
  {1, {127, 16}},
  {1, {127, 32}},
  {1, {127, 64}},
  {1, {127, 128}},

  {1, {437, 2}},
  {1, {437, 4}},
  {1, {437, 8}},
  {1, {437, 16}},
  {1, {437, 32}},
  {1, {437, 64}},
  {1, {437, 128}},

  {2, {1, 2, 2}},
  {2, {1, 4 ,4}},
  {2, {1, 8, 8}},
  {2, {1, 16, 16}},
  {2, {1, 32, 32}},
  {2, {1, 64, 64}},
  {2, {1, 128, 128}},

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
        table.insert(freqSize, math.floor(timeSize[i] / 2) + 1)
      else
        table.insert(freqSize, timeSize[i])
      end
    end
    table.insert(freqSize, 2)

    local timeInvCuda = timeCuda:clone()
    local frequencyCuda =
        torch.CudaTensor(torch.LongStorage(freqSize)):fill(0 / 0)

    local batchDims = #timeSize - fftDim
    local net = nn.FFTWrapper("cufft", 0, 0, "timed") -- no padding
    net:fft(timeCuda, frequencyCuda, batchDims)
    net:ffti(timeInvCuda, frequencyCuda, batchDims)

    local timeInv = timeInvCuda:double()
    local frequency = frequencyCuda:double()
    if printResults then
        print('cufft forward re:', frequency:select(fftDim + 2, 1))
        print('cufft forward im:', frequency:select(fftDim + 2, 2))
        print('cufft inverse re:', timeInv)
    end

    timeInvCuda = {}
    frequencyCuda = {}
    collectgarbage()

    return frequency, timeInv
end

local function transposedLayout(fftDim, fftSize)
   if fftDim == 2 and (fftSize < 8 or fftSize > 32) then return true end
   return false
end

local function benchmarkFBFFT(problemSize, timeCuda, frequency2)
    local fftDim   = problemSize[1]
    local timeSize = problemSize[2]
    local freqSize = {}
    local hermitianDim = #timeSize - fftDim + 1

    for i = 1, #timeSize do
        if i == hermitianDim then
            table.insert(freqSize, math.floor(timeSize[i] / 2) + 1)
        else
            table.insert(freqSize, timeSize[i])
        end
    end
    table.insert(freqSize, 2)

    local timeInvCuda = timeCuda:clone()
    local frequencyCuda =
        torch.CudaTensor(torch.LongStorage(freqSize)):fill(0 / 0)
    local batchDims = #timeSize - fftDim
    local net = nn.FFTWrapper("fbfft", 0, 0, "timed") -- no padding
    net:fft(timeCuda, frequencyCuda, batchDims)
    net:ffti(timeInvCuda, frequencyCuda, batchDims)

    local timeInv = timeInvCuda:double()
    local frequency = frequencyCuda:double()
    if transposedLayout(fftDim, timeSize[hermitianDim]) then
        frequency = frequency:transpose(2, 3)
    end

    if printResults then
        print('fbfft forward re:', frequency:select(fftDim + 2, 1))
        print('fbfft forward im:', frequency:select(fftDim + 2, 2))
        print('fb inverse re:', timeInv)
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
        local res = 0
        timeCudaTensor = torch.Tensor(
            torch.LongStorage(ps)):apply(function()
          val = val + 1
          if val == ps[#ps] + 1 then
              val = 1
              res = res + 1
          end
          return res
        end)
        if #timeCudaTensor:size() == 3 then
           timeCudaTensor = timeCudaTensor:transpose(2,3):contiguous():cuda()
        else
           timeCudaTensor = timeCudaTensor:cuda()
        end
    elseif localInit == 6 then
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

          local function checkEqual(a, b, complexCheck)
             if printResults then
                print('Top left block equality\n', a, b)
             end
             local a = a:double():abs() + 1e-10
             local b = b:double():abs() + 1e-10
             local delta = (a:double() - b:double()):abs()
             local max = a:max() + 1e-20
             local deltaNorm = delta:div(max)

             if printResults and deltaNorm:max() > precision then
                print('Check max delta, norm ref fft, max normalized, prec = ',
                      delta:max(), b:norm(), deltaNorm:max(), precision)
                print('RE:\n',
                      a:select(#a:size(), 1, 1),
                      b:select(#b:size(), 1, 1))
                print('IM:\n',
                      a:select(#a:size(), 2, 1),
                      b:select(#b:size(), 2, 1))
             end
             if deltaNorm:max() > precision then
                print('Error Delta RE', delta:select(#delta:size(), 1, 1))
                print('Error Delta IM', delta:select(#delta:size(), 2, 1))
             end
             assert(deltaNorm:max() <= precision,
                    deltaNorm:max() .. ' > ' .. precision)
          end

          local function checkOrthogonalSymmetry(r, fftSize)
             if printResults then
                print('Row orthogonal symmetry\n', r)
             end

             local max = r:clone():abs():max() + 1e-20
             for k = 1, fftSize / 2 - 1 do
                local d1 = r[fftSize / 2 + 1 - k][1] - r[fftSize / 2 + 1 + k][1]
                assert(
                   math.abs(d1) / max < precision,
                   d1 .. ' ' .. math.abs(d1) / max .. ' ' .. precision
                )
                local d2 = r[fftSize / 2 + 1 - k][2] + r[fftSize / 2 + 1 + k][2]
                assert(
                   math.abs(d2) / max < precision,
                   d2 .. ' ' .. math.abs(d2) / max .. ' ' .. precision
                )
             end
          end

          local
             function checkCentralSymmetry(cuFFT, fbFFT, fftSize, imaginaryPart)
                if printResults then
                   print('Remaining block central symmetry', cuFFT, fbFFT)
                end
                assert(cuFFT:size(1) == fbFFT:size(1))
                assert(cuFFT:size(2) == fbFFT:size(2))
                assert(cuFFT:size(3) == fbFFT:size(3))
                assert(cuFFT:size(2) == cuFFT:size(3))

                local max = cuFFT:clone():abs():max() + 1e-20
                for i = 1, cuFFT:size(1) do
                   for j = 1, cuFFT:size(2) do
                      for k = 1, cuFFT:size(2) do
                         local fbFFTVal = fbFFT
                            [i][1 + cuFFT:size(2) - j][1 + cuFFT:size(2) - k]

                         local d1 = cuFFT[i][j][k] - fbFFTVal
                         if imaginaryPart then
                            d1 = cuFFT[i][j][k] + fbFFTVal
                         end

                         if math.abs(d1) / max > precision then
                            print('Error Delta\n', d1, ' @ ', i, j, k)
                            print(cuFFT[i][j][k],
                                  ' vs ',
                                  fbFFTVal)
                         end
                         assert(
                            math.abs(d1) / max < precision,
                            d1 .. ' ' .. math.abs(d1) / max .. ' ' .. precision
                         )
                      end
                   end
                end
             end

          local function assertdiffHermitian(
                reffft, fbfft, fftDim, fftSize, complexCheck)
             if ps[1] > 512 then
                print('Skip long test based on lua side loops')
                return
             end

             if fftDim == 1 or (fftDim == 2 and not complexCheck) then
                -- Just check tensor relative equality modulo precision
                checkEqual(reffft:clone(), fbfft:clone())
             else
                assert(complexCheck)
                assert(fftDim == 2)
                -- Hermitian check is comprised of 4 checks, one is fbfft vs
                -- cufft, the others are symmetry checks
                checkEqual(
                   fbfft:narrow(
                      2, 1, fftSize / 2 + 1
                   ):narrow(3, 1, fftSize / 2 + 1):clone(),
                   reffft:narrow(
                      2, 1, fftSize / 2 + 1
                   ):narrow(3, 1, fftSize / 2 + 1):clone()
                )

                -- Orthogonal symmetry for first and middle rows along vertical
                -- plane FFTSize / 2 = 1
                for i = 1, reffft:size(1) do
                   for k = 1, fftSize / 2 + 1 do
                      checkOrthogonalSymmetry(fbfft[i][1]:clone(), fftSize)
                      checkOrthogonalSymmetry(
                         fbfft[i][fftSize / 2 + 1]:clone(), fftSize)
                   end
                end

                if fftSize > 2 then
                   -- Central symmetry for:
                   --   [1, FFTSize / 2) x [FFTSize / 2 + 1, FFTSize) and
                   --   [FFTSize / 2 + 1, FFTSize) x [FFTSize / 2 + 1, FFTSize)
                   local f = fbfft:narrow(
                      2, 2, (fftSize / 2 - 1)
                   ):narrow(
                      3, fftSize / 2 + 1 + 1, (fftSize / 2 - 1)
                           ):clone()
                   local c = reffft:narrow(
                      3, 2, (fftSize / 2 - 1)
                   ):narrow(
                      2, fftSize / 2 + 1 + 1, (fftSize / 2 - 1)
                           ):clone()
                   checkCentralSymmetry(
                      c:select(4, 1), f:select(4, 1), fftSize)
                   checkCentralSymmetry(
                      c:select(4, 2), f:select(4, 2), fftSize, true)
                end
             end
             return
          end

          local function assertdiffTransposed(reffft, fbfft, fftDim, fftSize)
             if ps[1] > 512 then
                print('Skip horrendously long test, need to transpose',
                      ' the data efficiently to test')
                return
             end
             local m = (reffft:double() - fbfft:double()):abs():max()
             local n = reffft:double():norm() + 1e-10
             local nfbfft = fbfft:double():norm() + 1e-10
             if m / n > precision then
                print('Check max diff, norm, norm fbfft, max normalized = ',
                      m, n, nfbfft, m / n)
                print('FAILS CHECK !!')
                print(m, n, m / n)
                if fftDim == 2 and #reffft:size() == 4 then
                   print('DIFFTENSOR REAL!\n')
                   print(reffft:add(-fbfft):float():select(fftDim + 2, 1))
                   print('DIFFTENSOR IM!\n')
                   print(reffft:add(-fbfft):float():select(fftDim + 2, 2))
                else
                   print(reffft, fbfft)
                   print('DIFFTENSOR REAL!\n')
                   print(reffft:add(-fbfft):float())
                end
             end
             assert(m / n < precision)
             return
          end

          local cufft, cuifft = benchmarkCuFFT(problemSizes[i], timeCudaTensor)
          local fbfft, fbifft =
              benchmarkFBFFT(problemSizes[i], timeCudaTensor, matchCuFFTAlloc)

          local fftSize = ps[2]
          if runTests then
             if not transposedLayout(fftDim, fftSize) then
                assertdiffHermitian(cufft, fbfft, fftDim, fftSize, true)
                assertdiffHermitian(cuifft, fbifft, fftDim, fftSize, false)
             else
                assertdiffTransposed(cufft, fbfft, fftDim, fftSize)
                assertdiffTransposed(cuifft, fbifft, fftDim, fftSize)
             end
          end

          timeCudaTensor = {}
          collectgarbage()
        end
    end
end

printResults = false
local localInits = {7} -- only run on random inputs to cut down testing time
local runCases = testCases

--[[
-- Convenient override of the default that are used for unit tests
localInits = {1}
runCases = _iclr2015TestCases
--]]

function FFTTester.test()
-- Type of initialization:
-- 1: fill(1.0f)
-- 2: 1.0f if 0 mod 2 else 2.0f
-- 3: val % 4 + 1
-- 4: val == row
-- 5: val == col
-- 6: starts at 1.0f and += 1.0f at each entry
-- else: random
   for i = 1, #localInits do
      run(localInits[i], runCases)
      collectgarbage()
      cutorch.synchronize()
   end
end

mytester:add(FFTTester)
mytester:run()

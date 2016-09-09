require 'fb.luaunit'
require 'cunn'
require 'fbcunn'
require 'nn'
require 'fbnn'

local precision = 1e-4
local threshold = 5e-5
local relaxedPrecision = 5 * 0.01668
local numRuns = 10
local benchmark = false
local debug = false
local silence = true
local seed = os.time()
print('Seed: ', seed)
math.randomseed(seed)
torch.manualSeed(seed)

local function BNTest(
      refmod, gpumod, input, gradOutput, debug, benchmark, indim)

   if debug then
      input:fill(1)
      gradOutput:fill(1)
      input:copy(torch.linspace(1, input:nElement(), input:nElement()))
      gradOutput:copy(torch.linspace(1, input:nElement(), input:nElement()))
   end

   -- batch norm without affine transform
   local function assertDiff(ref, actual, name)
      local rel, abs = nn.utils.relErr(ref, actual)
      if abs > threshold then
         assert(rel <= precision,
                name .. ' max diff ' .. ' absolute ' .. abs)
      else
         assert(rel <= relaxedPrecision,
                name .. ' max diff ' .. ' absolute ' .. abs)
      end
   end

   local function uniformInit(t1, t2)
      t1:uniform()
      t2:copy(t1)
   end

   for _, affine in ipairs({false, true}) do
      for _, train in ipairs({false, true}) do
         if not silence then
            print('affine, train', affine, train)
         end
         local modRef = refmod(indim, 1e-5, 0.1, affine):cuda()
         local modGPU = gpumod(indim, 1e-5, 0.1, affine)
         modGPU.train, modRef.train = train, train

         -- Preconditions
         if affine then
            -- Uniform both for testing purposes
            uniformInit(modRef.bias, modGPU.bias)
            uniformInit(modRef.weight, modGPU.weight)
            assertDiff(modRef.bias, modGPU.bias, 'bias')
            assertDiff(modRef.weight, modGPU.weight, 'weight')
         end
         -- uniformInit(modRef.running_std, modGPU.running_std)
         uniformInit(modRef.running_mean, modGPU.running_mean)
         -- assertDiff(modRef.running_std, modGPU.running_std, 'running_std')
         assertDiff(modRef.running_mean, modGPU.running_mean, 'running_mean')

         -- UpdateOutput
         modGPU:updateOutput(input)
         modRef:updateOutput(input)

         if debug then
            print('Input', input:float())
            print('GradOutput', gradOutput:float())
            print('weight', modGPU.weight:float())
            print('bias', modGPU.bias:float())
            print('Expected running_mean', modRef.running_mean:float())
            print('Actual running_mean', modGPU.running_mean:float())
            -- print('Expected running_std', modRef.running_std:float())
            -- print('Actual running_std', modGPU.running_std:float())
            print('Expected output', modRef.output:float())
            print('Actual output', modGPU.output:float())
         end

         -- Postconditions
         assertDiff(modRef.running_mean, modGPU.running_mean, 'running_mean')
         -- assertDiff(modRef.running_std, modGPU.running_std, 'running_std')
        assertDiff(modRef.output, modGPU.output, 'output')



         if train then
            -- Preconditions
            -- assertDiff(modRef.centered, modGPU.centered, 'centered')
            -- assertDiff(modRef.std, modGPU.std, 'std')
            if affine then
               assertDiff(modRef.weight, modGPU.weight, 'std')
            end

            -- UpdateGradInput
            modGPU:updateGradInput(input, gradOutput)
            modRef:updateGradInput(input, gradOutput)

            if debug then
               print('Expected gradInput', modRef.gradInput:float())
               print('Actual gradInput', modGPU.gradInput:float())
            end

            -- Postconditions
            assertDiff(modRef.gradInput, modGPU.gradInput, 'gradInput')

            if affine then
               -- Preconditions
               -- gradBias and gradWeight are unintialized, users usually
               -- call zeroGradParameters first, emulate this
               uniformInit(modRef.gradBias, modGPU.gradBias)
               uniformInit(modRef.gradWeight, modGPU.gradWeight)
               assertDiff(modRef.gradBias, modGPU.gradBias, 'gradBias')
               assertDiff(modRef.gradWeight, modGPU.gradWeight, 'gradWeight')
               -- assertDiff(modRef.normalized, modGPU.normalized, 'normalized')

               local scale = torch.random(1000) / 1000.0
               if debug then
                  local val = 0
                  gradOutput:apply(
                     function()
                        val = val + 1
                        return val
                     end
                  )
                  scale = 1.0
                  -- modRef.normalized:copy(modGPU.normalized)
                  -- print('Normalized', modRef.normalized:float())
                  print('GradOutput', gradOutput:float())
               end

               -- AccGradParameters
               modGPU:accGradParameters(input, gradOutput, scale)
               modRef:accGradParameters(input, gradOutput, scale)

               if debug then
                  print('Expected gradWeight', modRef.gradWeight:float())
                  print('Actual gradWeight', modGPU.gradWeight:float())
                  print('Expected gradBias', modRef.gradBias:float())
                  print('Actual gradBias', modGPU.gradBias:float())
               end

               -- Postconditions
               assertDiff(modRef.gradBias, modGPU.gradBias, 'gradBias')
               assertDiff(modRef.gradWeight, modGPU.gradWeight, 'gradWeight')
            end
         end
      end
   end
end

function testSpatialBatchNormalization()
   for i = 1, numRuns do
      local nframes, indim, ini, inj = torch.random(1, 17),
      torch.random(1, 19),
      torch.random(1, 35),
      torch.random(1, 35)
      if benchmark then
         nframes, indim, ini, inj = 128, 64, 112, 112
      end
      if debug then
         nframes, indim, ini, inj = 1, 1, 5, 7
      end

      local input = torch.zeros(nframes, indim, ini, inj):uniform():cuda()
      local gradOutput = torch.zeros(nframes, indim, ini, inj):uniform():cuda()

      BNTest(nn.SpatialBatchNormalization,
             fbnn.SpatialBatchNormalization,
             input,
             gradOutput,
             debug,
             benchmark,
             indim)
   end
end

function testBatchNormalization()
   for i = 1, numRuns do
      local nframes, indim = torch.random(1, 17), torch.random(1, 19)
      if benchmark then
         nframes, indim = 128, 4096
      end
      if debug then
         nframes, indim = 5, 7
      end

      local input = torch.zeros(nframes, indim):uniform():cuda()
      local gradOutput = torch.zeros(nframes, indim):uniform():cuda()

      BNTest(nn.BatchNormalization,
             fbnn.BatchNormalization,
             input,
             gradOutput,
             debug,
             benchmark,
             indim)
   end
end

--[[
 precision = 1e-6
 numRuns = 10
 benchmark = false
 debug = false
 silence = true
--]]

LuaUnit:main()

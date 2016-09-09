-- Copyright 2004-present Facebook. All Rights Reserved.

local ffi = require 'ffi'
local package_path = package.searchpath('cufft_wrapper', package.cpath)
if not package_path then -- not OSS
        package_path = 'torch_fb_fbcunn_cufft_wrapper'
end
local CuFFTFFI = ffi.load(package_path)

ffi.cdef[[
typedef int cufftHandle;
typedef int cufftResult;
typedef int cufftHandle;

typedef struct {
   cufftHandle handle;
} cufftHandleWrapper;

cufftResult cufftDestroy(cufftHandle plan);
void updateOutputBiasFFI(THCState*, THCudaTensor*, THCudaTensor*);
cufftHandle makeCuFFTPlanFFI(THCState* state,
                             THCudaTensor* realTH,
                             THCudaTensor* cplxTH,
                             bool direction,
                             bool normalize,
                             int fftVersion,
                             int batchDimensions);
]]

local FFTWrapper = torch.class('nn.FFTWrapper')

FFTWrapper.emptyBuffer = torch.CudaTensor()

function FFTWrapper:__init(cufft, padLeft, padUp, timed)
   self.batchDims = 0

   if cufft == nil or cufft == "cufft" then
      self.cufft = true
   else
      self.cufft = false
   end

   if timed == "timed" then
      self.timed = true
   else
      self.timed = false
   end

   self.padLeft = padLeft or 0
   self.padUp = padUp or 0
end

function FFTWrapper:fft(time, frequency, batchDims, plan)
   assert(batchDims >= 1)
   assert(batchDims <= 2)
   assert(torch.type(time) == 'torch.CudaTensor', 'FBFFT only with CudaTensors')
   self.batchDims = batchDims
   -- If calling fft from lua directly, just pass a buffer in any case.
   -- In practice it is only really needed for 2d-fft of size > 32
   local buffer = FFTWrapper.emptyBuffer
   if not self.cufft then
      -- Make full buffer to hold the whole complex tensor if needed
      -- TODO: Maybe fix this don't want to manage memory here.
      -- On the other hand we don't care much since we should use tiling anyway
      local fftDim = (#time:size() - batchDims)
      local needsBuffer = false
      for i = 1, fftDim do
         if time:size(self.batchDims + i) > 32 or
            frequency:size(self.batchDims + i) > 32 then
               needsBuffer = true
         end
      end
      if needsBuffer then
         if fbnn.SpatialConvolution.reportWarnings then
            print('FFTWrapper.lua: Perf killed by on-the-fly allocation, ',
                  'consider using tiling and stay under 32 FFT size')
         end
         buffer = frequency:clone()
      end
   end
   local handle = -1
   if plan then
      handle = plan.handle
   end
   time.nn.FFTWrapper_fft(self, time, frequency, buffer, handle)
end

function FFTWrapper:ffti(time, frequency, batchDims, plan)
   assert(batchDims >= 1)
   assert(batchDims <= 2)
   assert(torch.type(time) == 'torch.CudaTensor', 'FBFFT only with CudaTensors')
   self.batchDims = batchDims
   -- In practice it is only really needed for 2d-fft of size > 32
   local size = frequency:size()
   local buffer = FFTWrapper.emptyBuffer

   if not self.cufft then
      -- Make full buffer to hold the whole complex tensor if needed
      -- TODO: Maybe fix this don't want to manage memory here.
      -- On the other hand we don't care much since we should use tiling anyway
      local fftDim = (#time:size() - batchDims)
      local needsBuffer = false
      for i = 1, fftDim do
         if time:size(self.batchDims + i) > 32 or
            frequency:size(self.batchDims + i) > 32 then
               needsBuffer = true
         end
      end
      if needsBuffer and fftDim == 2 then
         if fbnn.SpatialConvolution.reportWarnings then
            print('FFTWrapper.lua: Perf killed by on-the-fly allocation, ',
                  'consider using tiling and stay under 32 FFT size')
         end
         if batchDims == 1 then
            local bufferSize = torch.LongStorage({
                  size[1], size[3], size[3], size[4]})
            buffer = torch.CudaTensor(bufferSize)
         elseif batchDims == 2 then
            local bufferSize = torch.LongStorage({
                  size[1], size[2], size[4], size[4], size[5]})
            buffer = torch.CudaTensor(bufferSize)
         end
      end
   end

   local handle = -1
   if plan then
      handle = plan.handle
   end

   time.nn.FFTWrapper_ffti(self, time, frequency, buffer, handle)
end


-- CuFFTPlan allocation occurs in here because it depends on the tensor shape
-- after transposition
function FFTWrapper:fftTranspose(tensor, bufferComplex, bufferComplexTranspose,
                                 batchDims, handle, stream, plan)
   local transposeSeparator = batchDims
   cutorch.setBlasHandle(handle)
   cutorch.setStream(stream)
   if self.cufft and not plan then
      local version = 0
      plan = ffi.new('cufftHandleWrapper')
      plan.handle = CuFFTFFI.makeCuFFTPlanFFI(cutorch._state,
                                              tensor:cdata(),
                                              bufferComplex:cdata(),
                                              true,
                                              false,
                                              version,
                                              batchDims)
      ffi.gc(plan, function(p)
                CuFFTFFI.cufftDestroy(p.handle)
      end)
   end
   self:fft(tensor, bufferComplex, batchDims, plan)
   local cublasWrapper = nn.CuBLASWrapper()
   cublasWrapper:transposeComplex(bufferComplex,
                                  bufferComplexTranspose,
                                  transposeSeparator,
                                  false,
                                  handle,
                                  stream)
   return plan
end

-- CuFFTPlan allocation occurs in here because it depends on the tensor shape
-- after transposition
function FFTWrapper:transposeIFFT(tensor, bufferComplex, bufferComplexTranspose,
                                  batchDims, handle, stream, plan)
   local transposeSeparator = batchDims
   cutorch.setBlasHandle(handle)
   cutorch.setStream(stream)
   local cublasWrapper = nn.CuBLASWrapper()
   cublasWrapper:transposeComplex(bufferComplexTranspose,
                                  bufferComplex,
                                  transposeSeparator,
                                  false,
                                  handle,
                                  stream)

   if self.cufft and not plan then
      local version = 0
      plan = ffi.new('cufftHandleWrapper')
      plan.handle = CuFFTFFI.makeCuFFTPlanFFI(cutorch._state,
                                              tensor:cdata(),
                                              bufferComplex:cdata(),
                                              false,
                                              false,
                                              version,
                                              batchDims)
      ffi.gc(plan, function(p)
                CuFFTFFI.cufftDestroy(p.handle)
      end)
   end
   self:ffti(tensor, bufferComplex, batchDims, plan)
   return plan
end

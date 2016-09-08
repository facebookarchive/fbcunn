local ffi = require 'ffi'

ffi.cdef[[
   void updateOutputBiasFFI(THCState*, THCudaTensor*, THCudaTensor*);
   void accGradParametersBiasFFI(
      THCState*, THCudaTensor*, THCudaTensor*, float scale);
   void transposeMMFFI(THCState*,
                       THCudaTensor* tA,
                       THCudaTensor* tB,
                       THCudaTensor* tC,
                       float invNorm,
                       bool conjugateTransposeA,
                       bool conjugateTransposeB,
                       bool accumulate);
   typedef struct {
      static const int FFT_UpdateOutput = 0;
      static const int FFT_UpdateGradInput = 1;
      static const int FFT_AccGradParameters = 2;
      int pass;
   } FFTConvolutionPassFFI;
   typedef struct {
     THCudaTensor* tensor;
     int padL;
     int padU;
   } TiledDeviceTensorFFI;
   void convolveIteratedFFI(THCState* state,
                            TiledDeviceTensorFFI* input,
                            THCudaTensor* weight,
                            TiledDeviceTensorFFI* output,
                            int numTiles,
                            int fftSize,
                            FFTConvolutionPassFFI pass,
                            float scale);
]]

// Copyright 2004-present Facebook. All Rights Reserved.

#include "cuda/Complex.cuh"
#include "cuda/ComputeCapabilities.cuh"
#include "cuda/CudaUtils.cuh"
#include "cuda/DeviceTensor.cuh"
#include "fft/CuFFTWrapper.cuh"
#include "THCTensor.h"
#include "DeviceTensorUtils.h"

#include <cuda_runtime.h>
#include <glog/logging.h>

using namespace facebook::cuda;

namespace facebook { namespace deeplearning { namespace torch {

namespace detail {

#define PI 3.14159265358979323846264338327f

__device__ __forceinline__
unsigned int reverse(unsigned int x, unsigned int nbits) {
  return __brev(x) >> (32 - nbits);
}

// This adjustment modulo FFTSize is used as a stepping stone to cram multiple
// FFTs of size < 32 into a single warp.
// The invariant is:
//   assert(FFTPerWarp * FFTSize == blockDim.x || FFTPerWarp == 1);
// This has no effect if FFTSize >= 32 or FFTPerWarp == 1.
// This is for the cases 2, 4, 8 and 16 and buys us additional perf.
template <int FFTSize>
__device__ __forceinline__ int adjustedThreadIdxX() {
  if (FFTSize < 32) {
    return (threadIdx.x & (FFTSize - 1));
  } else {
    return threadIdx.x;
  }
}

template <int FFTSize>
__device__ __forceinline__ int adjustedThreadIdxY() {
  if (FFTSize < 32) {
    return (threadIdx.y & (FFTSize - 1));
  } else {
    return threadIdx.y;
  }
}

// Computes the batch number based on the fact that batches are divided by:
//   - blockIdx.x, each block computes a chunk of bacthes,
//   - threadIdx.z, each z dimensions computes a subchcunk of batches to
//     increase occupancy,
//   - exactly FFTPerWarp FFTs are processed by one warp
// These 3 subdivisions interact to compute the actual batch size.
template <int FFTSize, int FFTPerWarp>
__device__ __forceinline__ int adjustedBatch() {
  if (FFTSize < 32) {
    int LogFFTSize = getMSB<FFTSize>();
    int LogFFTPerWarp = getMSB<FFTPerWarp>();
    return (threadIdx.x >> LogFFTSize) +
      (blockIdx.x << LogFFTPerWarp) +
      ((threadIdx.z * gridDim.x) << LogFFTPerWarp);
  } else {
    return blockIdx.x + threadIdx.z * gridDim.x;
  }
}

template <int FFTSize>
struct FFT1DCoeffs {
  enum {
    RegisterPerWarp = (FFTSize + WARP_SIZE - 1) / WARP_SIZE
  };
  __device__ __forceinline__ Complex& operator[](int i) {
    return coeff[i];
  }
  __device__ __forceinline__ Complex operator[](int i) const {
    return coeff[i];
  }

  Complex coeff[RegisterPerWarp];
};

static const int kNumTwiddles = 256;
__constant__ Complex twiddles[kNumTwiddles];

template <int FFTSize>
struct FFT1DRoots : public FFT1DCoeffs<FFTSize> {
  // Computes the twiddles for the least amount possible of registers and uses
  // trigonometric symmetries to populate the other registers.
  // We always compute at least 1 warpful of value using sincos.
  // For FFTs <= 32 we are done
  // For FFTs >= 32, given the number of registers per warp we know which
  // register indices fall at PI/4, PI/2, PI and 2*PI.
  // Since we always compute at least 1 warpful of values, we only consider
  // exact subdivisions of WARP_SIZE for symmetries.
  // For instance:
  //   - for FFTSize == 64, we have 2 registers corresponding to each half of
  //     the unit circle. We compute the first register (and not less by
  //     construction) and then we can use symmetry wrt -PI to fill the other
  //     register.
  //   - for FFTSize == 128, we have 4 registers corresponding to each
  //     quadrant of the unit circle. We compute the first register (and not
  //     less by construction) and then we can use symmetry wrt -PI/2 and -PI
  //     to fill the other registers.
  //
  // This is critical performance-wise and works well atm with unrolling.
  //
  // Twiddles are more efficiently computed for 1D FFTs and more efficiently
  // loaded from constant memory for 2D FFTs.
  __device__ __forceinline__ void twiddles1D() {
    // Note that we ever only need half the twiddles; see ASCII diagram:
    // for FFT-256 we only use w^0 .. w^127 and then recursively only halves.
    if (this->RegisterPerWarp >= 4) {
#pragma unroll
      for (int index = 0; index < this->RegisterPerWarp / 2; ++index) {
        // Can always use adjustedThreadIdxX since blockDim.x == WARP_SIZE
        // is enforced
        int x = adjustedThreadIdxX<FFTSize>() + index * WARP_SIZE;
        if (index < ceil((int)this->RegisterPerWarp, 4)) {
          // Compute in any case
          (*this)[index].sincos(-2.0f * PI * (1.0f / (float)FFTSize) * x);
        } else if (index < ceil((int)this->RegisterPerWarp, 2)) {
          // Symmetry wrt -PI/2
          (*this)[index] =
            (*this)[index - ceil((int)this->RegisterPerWarp, 4)]
            .transpose()
            .conjugate();
        } else {
          // Symmetry wrt -PI
          (*this)[index] = -(*this)[this->RegisterPerWarp - index];
        }
      }
    } else if (this->RegisterPerWarp == 2) {
      // Compute in any case, can always use adjustedThreadIdxX since
      // blockDim.x == WARP_SIZE is enforced
      int x = adjustedThreadIdxX<FFTSize>();
      (*this)[0].sincos(-2.0f * PI * (1.0f / (float)FFTSize) * x);
      // Symmetry wrt -PI, skip since only need half
    } else {
      // Compute in any case
      // adjustedThreadIdxX<FFTSize>() lets us cram multiple < 32 FFTs in
      // a warp
      int x = adjustedThreadIdxX<FFTSize>();
      (*this)[0].sincos(-2.0f * PI * (1.0f / (float)FFTSize) * x);
    }
  }

  __device__ __forceinline__ void twiddlesFromConstant1D() {
#pragma unroll
    for (int index = 0; index < this->RegisterPerWarp / 2; ++index) {
      int x = getLaneId() + index * WARP_SIZE;
      (*this)[index] = twiddles[x * (kNumTwiddles / FFTSize)];
    }
  }

};

template <int FFTSize>
struct FFT1DBitReversal {
  enum {
    RegisterPerWarp = (FFTSize + WARP_SIZE - 1) / WARP_SIZE
  };
  __device__ __forceinline__ int& operator[](int i) {
    return bitReversed[i];
  }
  __device__ __forceinline__ int operator[](int i) const {
    return bitReversed[i];
  }

  __device__ __forceinline__ void computeBitReversal(const int index) {
    int LogFFTSize = cuda::getMSB<FFTSize>();
    int x = adjustedThreadIdxX<FFTSize>() + index * blockDim.x;
    bitReversed[index] = reverse(x, LogFFTSize);
  }

  int bitReversed[RegisterPerWarp];
};

// Pure within a warp reversal for FFT sizes <= 32.
// For sizes >= 64 this is trickier since we need a cross-register,
// cross-warp bit reversal.
// Can be done inefficiently with a loop or local memory.
// Q: How can we make sure it will always unroll statically ?
// A: Just use shared memory for the bit reversal portion, it will only
// consume 2 * FFTSize floats per block.
template <int FFTSize, int FFTPerWarp>
 __device__ __forceinline__
void bitReverse1DWarp(FFT1DCoeffs<FFTSize>& coeffs,
                      const FFT1DBitReversal<FFTSize>& bits,
                      const int batch,
                      const int index) {
  assert(coeffs.RegisterPerWarp == 1);
  assert(index == 0);
  assert(FFTSize <= WARP_SIZE);

  // Only reverse and permute within blockDim.x boundary which allows to cram
  // multiple FFTs smaller than 32 into a single warp
  int LogFFTPerWarp = cuda::getMSB<FFTPerWarp>();
  coeffs[index] = shfl(coeffs[index],
                       bits[index],
                       blockDim.x >> LogFFTPerWarp);
}

// Helper function useful for maintaining the twiddle factor distribution
// invariant. Assuming registers r1 and r2, distributed across warps,
// we write r1[0, ... 31] and r2[0, ... 31].
// This concatenates r1 | r2 and keeps only the entries from the even warps.
// r1 and r2 both contain these values on exit.
// This is useful for simplifying the distribution of twiddle factors.
//
// Consider the case FFT-128, by construction:
//   r1[0, .. 31] == r3[0, .. 31] = [w^0 , .. w^31]
//   r2[0, .. 31] == r4[0, .. 31] = [w^32, .. w^63]
//
// After selectEvenWarpDistributed, all registers are equal and we have:
//   r1[0, .. 31] == ... == r4[0, .. 31] == [w^0, w^2, .. w^62]
//
// This occurs one more time to obtain:
//   r1[0, .. 31] == ... == r4[0, .. 31] == [w^0, w^4, .. w^60, 16 x garbage]
//
// The garbage is never read in decimateInFrequency1D32.
//
// Formally:
// r1[k] <- concat(r1, r2) [2k] for k \in [0 .. WARP_SIZE - 1]
// r2 <- r1
//
__device__ __forceinline__
void selectEvenWarpDistributed(Complex& r1, Complex& r2) {
  // E.g. stating from:
  //   r1[w^0, w^1, ... w^31] and r2[w^32, w^33, ...w^63]
  //
  // Set
  //   r1[w^0 , w^2 , ... w^30 |         16 x garbage]
  //   r2[16 x garbage         | w^32, w^34, ... w^62]
  //
  // And merge into:
  //   r1[w^0 , w^2 , ... w^30 | w^32, w^34, ... w^62]
  //
  // Dark compiler magic: trying to reduce this down to Complex loses 10%
  // perf. This seems related to instruction mix, divergence and the compiler
  // not able to reorder instructions past divergent points (which is
  // reasonable).
  r1.re() = shfl(r1.re(), 2 * getLaneId());
  r2.re() = shfl(r2.re(), 2 * getLaneId() - WARP_SIZE);
  if (threadIdx.x >= HALF_WARP_SIZE) {
    r1.re() = r2.re();
  }
  r1.im() = shfl(r1.im(), 2 * getLaneId());
  r2.im() = shfl(r2.im(), 2 * getLaneId() - WARP_SIZE);
  if (threadIdx.x >= HALF_WARP_SIZE) {
    r1.im() = r2.im();
  }
  r2 = r1;
}

template <int FFTSize>
__device__ __forceinline__ void load1D(const DeviceTensor<float, 2>& real,
                                       const DeviceTensor<float, 3>& complex,
                                       FFT1DCoeffs<FFTSize>& coeffs,
                                       FFT1DBitReversal<FFTSize>& bits,
                                       const int batch,
                                       const int index) {
  int LogFFTSize = getMSB<FFTSize>();
  // adjustedThreadIdxX<FFTSize>() lets us cram multiple < 32 FFTs in a warp
  int x = adjustedThreadIdxX<FFTSize>() + index * blockDim.x;
  bits[index] = reverse(x, LogFFTSize);

  // Support zero padding without a need to copy the input data to a larger
  // array.
  // TODO: center the kernel wrt to zeros.
  // TODO: support reflection padding: pass the kernel size to fill with
  // reflection and then zero after that to pad till the FFT size.
  // TODO: support complex input (just read the imaginary part)
  // TODO: try to do something with float4 and shuffles
  coeffs[index] =
    Complex((x < real.getSize(1)) ? real[batch][x].ldg() : 0.0f,
            0.0f);
}

template <int FFTSize>
__device__ __forceinline__ void store1D(DeviceTensor<float, 2>& real,
                                        DeviceTensor<float, 3>& complex,
                                        const FFT1DCoeffs<FFTSize>& coeffs,
                                        const int batch,
                                        const int index) {
  // adjustedThreadIdxX<FFTSize>() lets us cram multiple < 32 FFTs in a warp
  int x = adjustedThreadIdxX<FFTSize>() + index * blockDim.x;
  if (x < complex.getSize(1)) {
    // TODO: try to do something with float4 and shuffles
    *(complex[batch][x].dataAs<Complex>()) = coeffs[index];
  }
}

template <int FFTSize, int FFTSizeAlloc>
__device__ __forceinline__
void decimateInFrequency1D32(FFT1DCoeffs<FFTSizeAlloc>& coeffs,
                          const FFT1DRoots<FFTSizeAlloc>& roots,
                          const int index) {

  // Cannot be static due to upstream mix of function calls
  assert(FFTSize <= WARP_SIZE);
  assert(index < coeffs.RegisterPerWarp);

  int LogFFTSize = getMSB<FFTSize>();

#pragma unroll
  for (int logStep = 1; logStep <= LogFFTSize; ++logStep) {
    // Illustration for 1-D FFT of size 8, radix-2, decimation in frequency
    // Step 1 amongst 2,
    // Step 2 amongst 4,
    // Step 4 amongst 8,
    // ...

    Complex otherCoeff = shfl_xor(coeffs[index],
                                  (FFTSize >> logStep),
                                  (FFTSize >> (logStep - 1)));

    // Illustration for 1-D FFT of size 8, radix-2, decimation in frequency
    // Vals {1} U {3} U {5} U {7} amongst 2,
    // Vals [2, 3] U [6, 7] amongst 4,
    // Vals [4, 7] amongst 8,
    // ...
    otherCoeff = (threadIdx.x & (FFTSize >> logStep)) ?
      otherCoeff - coeffs[index] : coeffs[index] + otherCoeff;

    if (logStep < LogFFTSize) {
      // Illustration for 1-D FFT of size 8, radix-2, decimation in frequency
      // Twiddles [w^0, [w^0], w^0, [w^0], w^0, [w^0], w^0, [w^0]] amongst 2,
      // Twiddles [w^0, w^0, [w^0, w^2], w^0, w^0, [w^0, w^2]] amongst 4,
      // Twiddles [w^0, w^0, w^0, w^0, [w^0, w^1, w^2, w^3]] amongst 8,
      // ...
      int twiddleDee = (!(threadIdx.x & (FFTSize >> logStep))) ?
        0 : ((threadIdx.x & ((FFTSize >> logStep) - 1)) << (logStep - 1));
      Complex otherRoot = shfl(roots[index], twiddleDee);
      coeffs[index] = otherCoeff * otherRoot;
    } else {
      // Last step just does radix-2 + / - which is what otherCoeff contains
      coeffs[index] = otherCoeff;
    }
  }
}

template <int FFTSize>
struct TwiddleRebalancer {
  static __device__ __forceinline__
  void rebalance(FFT1DRoots<FFTSize>&, int);
};

template <> struct TwiddleRebalancer<64> {
  static __device__ __forceinline__
  void rebalance(FFT1DRoots<64>& roots, int) {
    selectEvenWarpDistributed(roots[0], roots[1]);
  }
};

template <> struct TwiddleRebalancer<128> {
  static __device__ __forceinline__
  void rebalance(FFT1DRoots<128>& roots, int logStep) {
    if (logStep == 1) {
      selectEvenWarpDistributed(roots[0], roots[1]);
      selectEvenWarpDistributed(roots[2], roots[3]);
      roots[1] = roots[2];
      roots[2] = roots[0];
    } else {
      assert(logStep == 2);
      selectEvenWarpDistributed(roots[0], roots[1]);
      roots[2] = roots[0];
      roots[3] = roots[0];
    }
  }
};

template <> struct TwiddleRebalancer<256> {
  static __device__ __forceinline__
  void rebalance(FFT1DRoots<256>& roots, int logStep) {
    if (logStep == 1) {
      selectEvenWarpDistributed(roots[0], roots[1]);
      selectEvenWarpDistributed(roots[2], roots[3]);
      selectEvenWarpDistributed(roots[4], roots[5]);
      selectEvenWarpDistributed(roots[6], roots[7]);
      roots[1] = roots[2];
      roots[2] = roots[4];
      roots[3] = roots[6];

      roots[4] = roots[0];
      roots[5] = roots[1];
      roots[6] = roots[2];
      roots[7] = roots[3];
    } else if (logStep == 2) {
      assert(logStep == 2);
      selectEvenWarpDistributed(roots[0], roots[1]);
      selectEvenWarpDistributed(roots[2], roots[3]);

      roots[1] = roots[2];

      roots[2] = roots[0];
      roots[3] = roots[1];

      roots[4] = roots[0];
      roots[5] = roots[1];
      roots[6] = roots[0];
      roots[7] = roots[1];
    } else {
      assert(logStep == 3);
      selectEvenWarpDistributed(roots[0], roots[1]);

      roots[1] = roots[0];

      roots[2] = roots[0];
      roots[3] = roots[0];

      roots[4] = roots[0];
      roots[5] = roots[0];
      roots[6] = roots[0];
      roots[7] = roots[0];
    }
  }
};

// The following ASCII shows the breakdown of a 1-D FFT-256 into
// the size 128 and 64-steps.
// Each 64 step is followed by 2 32-steps.
// A 32 step is the granularity of distributed storage (each warp holding 1
// value per 32-step).
// At this granularity, communication is exclusively across registers.
// Twiddle factors are continuously readjusted at each step.
// |-------|                |-------|
// | Reg0  |                | Reg0  |
// |       |                |-------|
// |-------|                | Reg1  |
// | Reg1  |                |-------|
// |-------|                |-------| w^0
// | Reg2  |                | Reg2  |  .
// |-------|                |-------|  .
// | Reg3  |                | Reg3  |  .
// |-------|                |-------| w^126 (increment 2)
//
// |-------|  w^0           |-------|
// | Reg4  |                | Reg4  |
// |       |                |-------|
// |-------|                | Reg5  |
// | Reg5  |   .            |-------|
// |-------|   .            |-------| w^0
// | Reg6  |   .            | Reg6  |  .
// |-------|                |-------|  .
// | Reg7  |                | Reg7  |  .
// |-------|  w^127 (+= 1)  |-------| w^126 (increment 2)
//
// E.g. for FFTSize = 256, we have 3 logSteps:
//   the first with 8 registers:
//     registers {{0, 4}, {1, 5}, {2, 6}, {3, 7}} communicate
//   the second with 4 registers:
//     registers {{0, 2}, {1, 3}, {4, 6}, {5, 7}} communicate
//   the third with 2 register
//     registers {{0, 1}, {2, 3}, {4, 5}, {6, 7}} communicate
//
// Note that everything is properly aligned modulo 32 and we don't need warp
// shuffles at all. The only exception may be the bit reversal phase which
// is currently implemented fully in shared memory since it would require
// fully unrolled, cross-register twiddles.
//
template <int FFTSize, int BatchUnroll>
__device__ __forceinline__
void decimateInFrequency1D(DeviceTensor<float, 2>& real,
                          DeviceTensor<float, 3>& complex,
                          FFT1DCoeffs<FFTSize>& coeffs,
                          const int batch) {
  // Cannot be static due to upstream mix of function calls
  assert(FFTSize >= WARP_SIZE);
  assert(blockDim.x == WARP_SIZE);

  int LogFFTSize = getMSB<FFTSize>();

  FFT1DBitReversal<FFTSize> bits;
#pragma unroll
  for (int i = 0; i < coeffs.RegisterPerWarp; ++i) {
    load1D<FFTSize>(real, complex, coeffs, bits, batch, i);
  }
  FFT1DRoots<FFTSize> roots;
  roots.twiddles1D();

  assert(coeffs.RegisterPerWarp == 1 << (LogFFTSize - LOG_WARP_SIZE));
  const int kDeltaLog = LogFFTSize - LOG_WARP_SIZE;
  {
    // Computation is all within the same warp across registers.
    // Unlike shuffles, things do not update in parallel so we do have
    // WAR (a.k.a false) dependences -> need a swap temporary storage !
    // Make swap registers local to this scope
    FFT1DCoeffs<FFTSize> swap;
#pragma unroll
    for (int logStep = 1; logStep <= kDeltaLog; ++logStep) {
      // Always need to process all the registers, this is not a function of
      // the logStep but only of the coeffs.RegisterPerWarp.
      // The spacing between registers that communicate is however a function
      // of logStep.
#pragma unroll
      for (int reg = 0; reg < coeffs.RegisterPerWarp; ++reg) {
        // By how many registers are we stepping ?
        // e.g. LogFFTSize == 8, LOG_WARP_SIZE == 5, logStep == 1 ->
        //   kDeltaLog == 3, kDeltaStep = 4
        const int kDeltaStep = (1 << (kDeltaLog - logStep));
        assert(kDeltaStep >= 0);
        assert(kDeltaStep < coeffs.RegisterPerWarp);

        // If bit kDeltaStep is step then sub else add
        int reg2 = (reg & kDeltaStep) ? reg - kDeltaStep : reg + kDeltaStep;
        // Sanity check
        assert(reg != reg2);

        Complex otherCoeff = coeffs[reg2];
        otherCoeff = (reg > reg2) ?
          otherCoeff - coeffs[reg] : coeffs[reg] + otherCoeff;

        // Only second half requires twiddling
        if (reg > reg2) {
          // Enforce this invariant:
          //   the register is exactly reg2 and no shuffle necessary until <= 32
          Complex otherRoot = roots[reg2];
          // Here we could write directly to vals and not swap but performance
          // is higher writing swap, likely due to same register writing
          // across branches and predicated code generated by the compiler.
          swap.coeff[reg] = otherCoeff * otherRoot;
        } else {
          swap.coeff[reg] = otherCoeff;
        }
      }

      // Recover values from swap
#pragma unroll
      for (int reg = 0; reg < coeffs.RegisterPerWarp; ++reg) {
        coeffs[reg] = swap.coeff[reg];
      }

      // This piece of code serves the purpose of rebalancing the twiddle
      // factors across registers within a warp by merging 2 consecutive
      // registers and selecting the odd entries (effectively keeping:
      //   w^0, w^2 ... w^2*(N/2) out of w^0, w^1, ... w^N).
      // Once this is done, we have something like:
      //   w^0 .. w^62 | garbage | w^64 .. w^128 | garbage
      // That needs to be copied into:
      //   w^0 .. w^62 | w^64 .. w^128 | w^0 .. w^62 | w^64 .. w^128
      //
      // In the general case, this has a recursive behavior with log-style RAW
      // / WAR dependencies.
      // It requires full unrolling or perf will die.
      // This is what limits the FFT size to 256 atm.
      // Cannot be static due to upstream mix of function calls
      assert(1 <= coeffs.RegisterPerWarp && coeffs.RegisterPerWarp <= 8);
      assert(32 <= FFTSize && FFTSize <= 256);
      // TODO: Figure out how to replace the monstruosity within
      TwiddleRebalancer<FFTSize>::rebalance(roots, logStep);
    }
  }

  // At this point we reached the FFT32, do them all in sequence
#pragma unroll
  for (int i = 0; i < (1 << kDeltaLog); ++i) {
    decimateInFrequency1D32<WARP_SIZE, FFTSize>(coeffs, roots, i);
  }

  {
    // Bit reversal through shared memory because double indirection is not
    // easily unrolled.
    // TODO: see if we can use float4
    // TODO: purely in registers, starting at 256 smem already gnaws at
    // occupancy.
    // No need to sync, dependences within a single warp
    __shared__ Complex buffer[BatchUnroll][FFTSize];
    assert(blockDim.z == BatchUnroll);
#pragma unroll
    for (int reg = 0; reg < coeffs.RegisterPerWarp; ++reg) {
      int x = getLaneId() + reg * WARP_SIZE;
      buffer[threadIdx.z][x] = coeffs[reg];
    }
    // No need to sync, dependences within a single warp
#pragma unroll
    for (int reg = 0; reg < coeffs.RegisterPerWarp; ++reg) {
      coeffs[reg] = buffer[threadIdx.z][bits[reg]];
    }
    // No need to sync, dependences within a single warp

#pragma unroll
    for (int reg = 0; reg < coeffs.RegisterPerWarp; ++reg) {
      store1D<FFTSize>(real, complex, coeffs, batch, reg);
    }
  }

}


template <int FFTSize, int BatchUnroll, int FFTPerWarp>
__global__ void decimateInFrequency1DKernel(DeviceTensor<float, 2> real,
                                            DeviceTensor<float, 3> complex) {
  // Ensure proper usage of the BatchUnroll template parameter which controls
  // static shared memory allocation for bit reversals of FFTs >= 64
  // TODO: default template parameter cuda-7
  cuda_static_assert((FFTSize > WARP_SIZE && BatchUnroll >= 1) ||
                     (FFTSize <= WARP_SIZE && BatchUnroll == 1));
  cuda_static_assert(!(FFTPerWarp & (FFTPerWarp - 1)));
  cuda_static_assert(FFTPerWarp * FFTSize <= WARP_SIZE ||
                     FFTPerWarp == 1);
  assert(FFTPerWarp * FFTSize == blockDim.x || FFTPerWarp == 1);

  int LogFFTSize = getMSB<FFTSize>();
  int LogFFTPerWarp = getMSB<FFTPerWarp>();

  // Enforce that the number of FFTs we perform is divisible by the number of
  // FFTs per warp, otherwise weird divergence will occur and possibly bugs.
  assert(real.getSize(0) % FFTPerWarp == 0);
  const int batch = adjustedBatch<FFTSize, FFTPerWarp>();
  if (batch >= real.getSize(0)) {
    return;
  }

  if (FFTSize <= 32) {
    FFT1DCoeffs<FFTSize> coeffs;
    FFT1DBitReversal<FFTSize> bits;
    load1D<FFTSize>(real, complex, coeffs, bits, batch, 0);
    FFT1DRoots<FFTSize> roots;
    roots.twiddles1D();
    decimateInFrequency1D32<FFTSize, FFTSize>(coeffs, roots, 0);
    bitReverse1DWarp<FFTSize, FFTPerWarp>(coeffs, bits, batch, 0);
    store1D<FFTSize>(real, complex, coeffs, batch, 0);
  } else {
    FFT1DCoeffs<FFTSize> coeffs;
    decimateInFrequency1D<FFTSize, BatchUnroll>(real, complex, coeffs, batch);
  }
}

template <int BatchDims>
FFTParameters::ErrorCode fbfft1D(
    DeviceTensor<float, BatchDims + 1>& real,
    DeviceTensor<float, BatchDims + 2>& complex) {
  // TODO: The limiter for size 256 is the twiddle cross-register shuffle
  // implementation that is currently unrolled by hand.
  // TODO: Starting 512, the occupancy goes down due to shared memory bit
  // reversal.
  assert(real.getSize(1) <= 256);
  assert(BatchDims == 1);

#define SELECT_FBFFT_1D_DIF_LE32(FFT_SIZE, BATCH_UNROLL, FFTS_PER_WARP) \
  if (real.getSize(1) == FFT_SIZE) {                                    \
    cuda_static_assert(FFT_SIZE <= 32);                                 \
    if (real.getSize(0) % FFTS_PER_WARP == 0) {                         \
      dim3 blocks(ceil(ceil(real.getSize(0), FFTS_PER_WARP),            \
                       BATCH_UNROLL));                                  \
      dim3 threads(real.getSize(1) * FFTS_PER_WARP, 1, BATCH_UNROLL);   \
      decimateInFrequency1DKernel<FFT_SIZE, 1, FFTS_PER_WARP>           \
        <<<blocks, threads>>>(real, complex);                           \
    } else {                                                            \
      dim3 blocks(ceil(real.getSize(0), BATCH_UNROLL));                 \
      dim3 threads(real.getSize(1), 1, BATCH_UNROLL);                   \
      decimateInFrequency1DKernel<FFT_SIZE, 1, 1><<<blocks, threads>>>( \
        real, complex);                                                 \
    }                                                                   \
    return FFTParameters::Success;                                      \
  }

#define SELECT_FBFFT_1D_DIF_GT32(FFT_SIZE, BATCH_UNROLL)        \
  if (real.getSize(1) == FFT_SIZE) {                            \
    cuda_static_assert(FFT_SIZE > 32);                          \
    dim3 blocks(ceil(real.getSize(0), BATCH_UNROLL));           \
    dim3 threads(32, 1, BATCH_UNROLL);                          \
    decimateInFrequency1DKernel<FFT_SIZE, BATCH_UNROLL, 1>      \
      <<<blocks, threads>>>(real, complex);                     \
    return FFTParameters::Success;                              \
  }

  SELECT_FBFFT_1D_DIF_LE32( 2, 32, 16);
  SELECT_FBFFT_1D_DIF_LE32( 4, 16,  8);
  SELECT_FBFFT_1D_DIF_LE32( 8,  8,  4);
  SELECT_FBFFT_1D_DIF_LE32(16,  4,  2);
  SELECT_FBFFT_1D_DIF_LE32(32,  4,  1);
  SELECT_FBFFT_1D_DIF_GT32(64,  4);
  SELECT_FBFFT_1D_DIF_GT32(128, 4);
  SELECT_FBFFT_1D_DIF_GT32(256, 2);

  return FFTParameters::UnsupportedSize;
}


template <int FFTSize>
__device__ __forceinline__ void load2D(const DeviceTensor<float, 3>& real,
                                       const DeviceTensor<float, 4>& complex,
                                       FFT1DCoeffs<FFTSize>& coeffs,
                                       const int batch,
                                       const int indexX,
                                       const int indexY) {
  int LogFFTSize = getMSB<FFTSize>();
  // adjustedThreadIdxX<FFTSize>() lets us cram multiple < 32 FFTs in a warp
  int x = adjustedThreadIdxX<FFTSize>() + indexX * blockDim.x;
  // adjustedThreadIdxX<FFTSize>() lets us cram multiple < 32 FFTs in a warp
  int y = adjustedThreadIdxY<FFTSize>() + indexY * blockDim.y;

  // Support zero padding without a need to copy the input data to a larger
  // array.
  // TODO: center the kernel wrt to zeros.
  // TODO: support reflection padding: pass the kernel size to fill with
  // reflection and then zero after that to pad till the FFT size.
  // TODO: support complex input (just read the imaginary part)
  // TODO: try to do something with float4 and shuffles
  coeffs[indexX] =
    Complex((y < real.getSize(1) && x < real.getSize(2)) ?
            real[batch][y][x].ldg() : 0.0f,
            0.0f);
}

template <int FFTSize>
__device__ __forceinline__ void store2D(DeviceTensor<float, 3>& real,
                                        DeviceTensor<float, 4>& complex,
                                        const FFT1DCoeffs<FFTSize>& coeffs,
                                        const int batch,
                                        const int indexX,
                                        const int indexY) {
  // adjustedThreadIdxX<FFTSize>() lets us cram multiple < 32 FFTs in a warp
  int x = adjustedThreadIdxX<FFTSize>() + indexX * blockDim.x;
  // adjustedThreadIdxX<FFTSize>() lets us cram multiple < 32 FFTs in a warp
  int y = adjustedThreadIdxY<FFTSize>() + indexY * blockDim.y;
  if (y < complex.getSize(1) && x < complex.getSize(2)) {
    // TODO: try to do something with float4 and shuffles
    *(complex[batch][y][x].dataAs<Complex>()) = coeffs[indexX];
  }
}

// Performs cross warp transpose of the data in registers, synchronously for
// each register at a time and takes advantage of Hermitian symmetry.
//
// Invariants are:
//  - not synchronized on entry of the loop
//  - synchronized at each step of the loop
//  - synchronized on exittemplate <int FFTSize, int BatchUnroll>
template <int FFTSize, int BatchUnroll, int SMemRows>
__device__ __forceinline__ void transpose2DHermitian(
  FFT1DCoeffs<FFTSize>& coeffsLo,
  FFT1DCoeffs<FFTSize>& coeffsHi,
  Complex(*buffer)[SMemRows][SMemRows / 2 + 1]) {
#pragma unroll
  for (int reg = 0; reg < coeffsLo.RegisterPerWarp; ++reg) {
    if (threadIdx.x < blockDim.x / 2 + 1) {
      buffer[threadIdx.z][threadIdx.y][threadIdx.x] = coeffsLo.coeff[reg];
      buffer[threadIdx.z][threadIdx.y + blockDim.y][threadIdx.x] =
        coeffsHi.coeff[reg];
    }
    __syncthreads();
    coeffsLo.coeff[reg] = buffer[threadIdx.z][threadIdx.x][threadIdx.y];
    if (threadIdx.y == 0) {
      coeffsHi.coeff[reg] =
        buffer[threadIdx.z][threadIdx.x][threadIdx.y + blockDim.y];
    }
    __syncthreads();
  }
}

// Performs cross warp transpose of the data in registers, synchronously for
// each register at a time and takes advantage of Hermitian symmetry.
//
// Invariants are:
//  - not synchronized on entry of the loop
//  - synchronized at each step of the loop
//  - synchronized on exittemplate <int FFTSize, int BatchUnroll>
template <int FFTSize, int BatchUnroll, int SMemRows>
__device__ __forceinline__ void untranspose2DHermitianOutput(
  FFT1DCoeffs<FFTSize>& coeffsLo,
  FFT1DCoeffs<FFTSize>& coeffsHi,
  Complex(*buffer)[SMemRows / 2 + 1][SMemRows + 1]) {
#pragma unroll
  for (int reg = 0; reg < coeffsLo.RegisterPerWarp; ++reg) {
    buffer[threadIdx.z][threadIdx.y][threadIdx.x] = coeffsLo.coeff[reg];
    if (threadIdx.y == 0) {
      buffer[threadIdx.z][threadIdx.y + blockDim.y][threadIdx.x] =
        coeffsHi.coeff[reg];
    }
    __syncthreads();
    if (threadIdx.x < blockDim.x / 2 + 1) {
      coeffsLo.coeff[reg] = buffer[threadIdx.z][threadIdx.x][threadIdx.y];
      coeffsHi.coeff[reg] =
        buffer[threadIdx.z][threadIdx.x][threadIdx.y + blockDim.y];
    }
    __syncthreads();
  }
}

// In the 2-D real to complex case, we can exploit Hermitian symmetry.
// We exploit the symmetry to cut in half the amount of work  for sizes >= 32.
// Given a square FFT of size NxN (power of 2), with Hermitian symmetry we
// only need to compute N x (N / 2 + 1) after transposition.
// The N / 2 + 1 factor is problematic because it typically results in sizes
// such as 32 x 17. This is a bad scenario for GPU occupancy.
// Instead, we implement this as 32 x 16 with a Lo and Hi register.
// Every threadIdx.y performs work on the Lo register but only threadIdx.y ==
// 0 performs work on the Hi register.
// This results in a much better occupancy and a 30% performance improvement.
template <int FFTSize, int BatchUnroll>
__global__ void decimateInFrequencyHermitian2D32Kernel(
    DeviceTensor<float, 3> real, DeviceTensor<float, 4> complex) {
  // Ensure proper usage of the BatchUnroll template parameter which controls
  // static shared memory allocation for bit reversals of FFTs >= 64
  // TODO: default template parameter cuda-7
  // cuda_static_assert((FFTSize > WARP_SIZE && BatchUnroll >= 1) ||
  //                    (FFTSize <= WARP_SIZE && BatchUnroll == 1));

  int LogFFTSize = getMSB<FFTSize>();
  // Enforce that the number of FFTs we perform is divisible by the number of
  // FFTs per warp, otherwise weird divergence will occur and possibly bugs.
  const int batch = adjustedBatch<FFTSize, 1>();
  if (batch >= real.getSize(0)) {
    return;
  }

  FFT1DCoeffs<FFTSize> coeffs;
  __shared__ Complex buffer[BatchUnroll][WARP_SIZE / 2 + 1][WARP_SIZE + 1];

  cuda_static_assert(FFTSize <= 32);

  FFT1DBitReversal<FFTSize> bits;
  bits.computeBitReversal(0);
  // Twiddles is the same as for 1D but fully data parallel across threadIdx.y
  FFT1DRoots<FFTSize> roots;
  roots.twiddles1D();

  FFT1DCoeffs<FFTSize> coeffsHi;
  FFT1DCoeffs<FFTSize> coeffsLo;
  load2D<FFTSize>(real, complex, coeffsLo, batch, 0, 0);
  load2D<FFTSize>(real, complex, coeffsHi, batch, 0, 1);
  decimateInFrequency1D32<FFTSize, FFTSize>(coeffsLo, roots, 0);
  decimateInFrequency1D32<FFTSize, FFTSize>(coeffsHi, roots, 0);
  bitReverse1DWarp<FFTSize, 1>(coeffsLo, bits, batch, 0);
  bitReverse1DWarp<FFTSize, 1>(coeffsHi, bits, batch, 0);

  transpose2DHermitian<FFTSize, BatchUnroll, WARP_SIZE>(
    coeffsLo,
    coeffsHi,
    (Complex(*)[WARP_SIZE][WARP_SIZE / 2 + 1])buffer);

  decimateInFrequency1D32<FFTSize, FFTSize>(coeffsLo, roots, 0);
  // Bit reversal is the same as for 1D but fully data parallel across
  // threadIdx.y
  bitReverse1DWarp<FFTSize, 1>(coeffsLo, bits, batch, 0);
  if (threadIdx.y == 0) {
    decimateInFrequency1D32<FFTSize, FFTSize>(coeffsHi, roots, 0);
    // Bit reversal is the same as for 1D but fully data parallel across
    // threadIdx.y
    bitReverse1DWarp<FFTSize, 1>(coeffsHi, bits, batch, 0);
  }

  untranspose2DHermitianOutput<FFTSize, BatchUnroll, WARP_SIZE>(
    coeffsLo,
    coeffsHi,
    (Complex(*)[WARP_SIZE / 2 + 1][WARP_SIZE + 1])buffer);

  store2D<FFTSize>(real, complex, coeffsLo, batch, 0, 0);
  store2D<FFTSize>(real, complex, coeffsHi, batch, 0, 1);
}

// Performs cross warp transpose of the data in registers, synchronously for
// each register at a time.
//
// Invariants are:
//  - not synchronized on entry of the loop
//  - synchronized at each step of the loop
//  - synchronized on exit
template <int FFTSize, int BatchUnroll, int SMemRows>
__device__ __forceinline__ void transpose2D(
  FFT1DCoeffs<FFTSize>& coeffs,
  Complex(*buffer)[SMemRows][SMemRows + 1]) {
#pragma unroll
  for (int reg = 0; reg < coeffs.RegisterPerWarp; ++reg) {
    buffer[threadIdx.z][threadIdx.y][threadIdx.x] = coeffs[reg];
    __syncthreads();
    coeffs[reg] = buffer[threadIdx.z][threadIdx.x][threadIdx.y];
    __syncthreads();
  }
}

template <int FFTSize, int BatchUnroll, int FFTPerWarp>
__global__ void decimateInFrequency2D32Kernel(DeviceTensor<float, 3> real,
                                 DeviceTensor<float, 4> complex) {
  // Ensure proper usage of the BatchUnroll template parameter which controls
  // static shared memory allocation for bit reversals of FFTs >= 64
  // TODO: default template parameter cuda-7
  // cuda_static_assert((FFTSize > WARP_SIZE && BatchUnroll >= 1) ||
  //                    (FFTSize <= WARP_SIZE && BatchUnroll == 1));
  cuda_static_assert(FFTSize < 32);
  cuda_static_assert(!(FFTPerWarp & (FFTPerWarp - 1)));
  cuda_static_assert(FFTPerWarp * FFTSize <= WARP_SIZE ||
                     FFTPerWarp == 1);
  assert(FFTPerWarp * FFTSize == blockDim.x || FFTPerWarp == 1);
  assert(blockDim.x == blockDim.y);

  int LogFFTSize = getMSB<FFTSize>();
  int LogFFTPerWarp = getMSB<FFTPerWarp>();
  // Enforce that the number of FFTs we perform is divisible by the number of
  // FFTs per warp, otherwise weird divergence will occur and possibly bugs.
  assert(real.getSize(0) % FFTPerWarp == 0);
  const int batch = adjustedBatch<FFTSize, FFTPerWarp>();
  if (batch >= real.getSize(0)) {
    return;
  }

  __shared__ Complex
    buffer[BatchUnroll][FFTSize * FFTPerWarp][FFTSize * FFTPerWarp + 1];

  cuda_static_assert(FFTSize <= 32);
  FFT1DCoeffs<FFTSize> coeffs;
  FFT1DBitReversal<FFTSize> bits;
  bits.computeBitReversal(0);
  load2D<FFTSize>(real, complex, coeffs, batch, 0, 0);
  // Twiddles is the same as for 1D but fully data parallel across threadIdx.y
  FFT1DRoots<FFTSize> roots;
  roots.twiddles1D();
  decimateInFrequency1D32<FFTSize, FFTSize>(coeffs, roots, 0);
  bitReverse1DWarp<FFTSize, FFTPerWarp>(coeffs, bits, batch, 0);
  transpose2D<FFTSize, BatchUnroll, FFTSize * FFTPerWarp>(coeffs, buffer);
  decimateInFrequency1D32<FFTSize, FFTSize>(coeffs, roots, 0);
  // Bit reversal is the same as for 1D but fully data parallel across
  // threadIdx.y
  bits.computeBitReversal(0);
  bitReverse1DWarp<FFTSize, FFTPerWarp>(coeffs, bits, batch, 0);
  transpose2D<FFTSize, BatchUnroll, FFTSize * FFTPerWarp>(coeffs, buffer);
  store2D<FFTSize>(real, complex, coeffs, batch, 0, 0);
}

template <int BatchDims>
FFTParameters::ErrorCode fbfft2D(DeviceTensor<float, BatchDims + 2>& real,
                                 DeviceTensor<float, BatchDims + 3>& complex) {
  // TODO: The limiter for size 256 is the twiddle cross-register shuffle
  // implementation that is currently unrolled by hand.
  // TODO: Starting 512, the occupancy goes down due to shared memory bit
  // reversal.
  if (real.getSize(1) != real.getSize(2) || real.getSize(1) > 256) {
    return FFTParameters::UnsupportedSize;
  }
  if (BatchDims != 1) {
    return FFTParameters::UnsupportedDimension;
  }

#define SELECT_FBFFT_2D_DIF_MULTIPLE(FFT_SIZE, FFTS_PER_WARP, BATCH_UNROLL) \
  if (real.getSize(1) == FFT_SIZE) {                                    \
    if (FFT_SIZE != real.getSize(2)) {                                  \
      return FFTParameters::UnsupportedSize;                            \
    }                                                                   \
    if (real.getSize(0) % FFTS_PER_WARP == 0) {                         \
      dim3 blocks(ceil(real.getSize(0), BATCH_UNROLL * FFTS_PER_WARP)); \
      dim3 threads(real.getSize(1) * FFTS_PER_WARP,                     \
                   real.getSize(2) * FFTS_PER_WARP,                     \
                   BATCH_UNROLL);                                       \
      decimateInFrequency2D32Kernel<FFT_SIZE, BATCH_UNROLL, FFTS_PER_WARP> \
        <<<blocks, threads>>>(real, complex);                           \
    } else {                                                            \
      dim3 blocks(ceil(real.getSize(0), BATCH_UNROLL));                 \
      dim3 threads(real.getSize(1),                                     \
                   real.getSize(2) / 2,                                 \
                   BATCH_UNROLL);                                       \
      decimateInFrequencyHermitian2D32Kernel<FFT_SIZE, BATCH_UNROLL>    \
        <<<blocks, threads>>>(real, complex);                           \
    }                                                                   \
    return FFTParameters::Success;                                      \
  }

#define SELECT_FBFFT_2D_DIF_SINGLE(FFT_SIZE, FFTS_PER_WARP, BATCH_UNROLL) \
  if (real.getSize(1) == FFT_SIZE) {                                    \
    if (FFT_SIZE != real.getSize(2)) {                                  \
      return FFTParameters::UnsupportedSize;                            \
    }                                                                   \
    dim3 blocks(ceil(real.getSize(0), BATCH_UNROLL));                   \
    dim3 threads(real.getSize(1),                                       \
                 real.getSize(2) / 2,                                   \
                 BATCH_UNROLL);                                         \
    decimateInFrequencyHermitian2D32Kernel<FFT_SIZE, BATCH_UNROLL>      \
      <<<blocks, threads>>>(real, complex);                             \
    return FFTParameters::Success;                                      \
  }

  // TODO: limit this size with cuda-7.
  // This really calls for a tight loop with constexpr
  SELECT_FBFFT_2D_DIF_MULTIPLE(2, 8, 1);
  SELECT_FBFFT_2D_DIF_MULTIPLE(4, 4, 4);
  SELECT_FBFFT_2D_DIF_MULTIPLE(8, 1, 4);
  SELECT_FBFFT_2D_DIF_SINGLE(16, 1, 1);
  SELECT_FBFFT_2D_DIF_SINGLE(32, 1, 1);

  return FFTParameters::UnsupportedSize;
}

} // detail

template <int Batch, int Dim>
FFTParameters::ErrorCode fbfft(THCudaTensor* r,
                               THCudaTensor* c,
                               FFTParameters params) {
  assert(params.fbFFT());
  if (Batch == 1 && Dim == 2) {
    DeviceTensor<float, 2> real = torchToDeviceTensorCast<float, 2>(r);
    DeviceTensor<float, 3> complex = torchToDeviceTensorCast<float, 3>(c);
    return detail::fbfft1D<Batch>(real, complex);
  } else if (Batch == 1 && Dim == 3) {
    DeviceTensor<float, 3> real = torchToDeviceTensorCast<float, 3>(r);
    DeviceTensor<float, 4> complex = torchToDeviceTensorCast<float, 4>(c);
    return detail::fbfft2D<Batch>(real, complex);
  } else {
    return FFTParameters::UnsupportedDimension;
  }
}

template FFTParameters::ErrorCode
fbfft<1, 2>(THCudaTensor* real, THCudaTensor* complex, FFTParameters params);

template FFTParameters::ErrorCode
fbfft<1, 3>(THCudaTensor* real, THCudaTensor* complex, FFTParameters params);

} } } // namespace

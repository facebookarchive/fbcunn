// Copyright 2004-present Facebook. All Rights Reserved.

#include "OneBitQuantization.cuh"

#include "cuda/ComputeCapabilities.cuh"
#include "cuda/CudaDebugUtils.cuh"
#include "cuda/CudaUtils.cuh"
#include "cuda/WarpReductions.cuh"

using namespace facebook::cuda;

#define NUM_BITS (int) (sizeof(unsigned) * 8)

namespace facebook { namespace deeplearning { namespace torch {

namespace detail {

__device__ __forceinline__ int getRow() {
  return blockIdx.x;
}

__global__ void
quantize(const DeviceTensor<float, 2> in,
         DeviceTensor<unsigned, 2> out,
         DeviceTensor<float, 2> quantizationError,
         DeviceTensor<float, 1> avgPos,
         DeviceTensor<float, 1> avgNeg) {
  // This code only works if this is the case
  assert(warpSize == WARP_SIZE && WARP_SIZE == NUM_BITS);

  // Calculate per-row averages
  const int row = getRow();

  float pos = 0.0f;
  int posCount = 0;
  float neg = 0.0f;
  int negCount = 0;

  for (int col = threadIdx.x; col < in.getSize(1); col += blockDim.x) {
    const float val = in[row][col] + quantizationError[row][col];
    if (val >= 0.0f) {
      pos += val;
      ++posCount;
    } else {
      neg += val;
      ++negCount;
    }
  }

  // Warp reduce the values
  pos = warpReduceSum(pos);
  posCount = warpReduceSum(posCount);
  neg = warpReduceSum(neg);
  negCount = warpReduceSum(negCount);

  // Block reduce the values, one for each warp
  __shared__ float blockPos[4];
  __shared__ int blockPosCount[4];
  __shared__ float blockNeg[4];
  __shared__ int blockNegCount[4];

  if (getLaneId() == 0) {
    blockPos[getWarpId()] = pos;
    blockPosCount[getWarpId()] = posCount;
    blockNeg[getWarpId()] = neg;
    blockNegCount[getWarpId()] = negCount;
  }

  __syncthreads();

  // We need all threads in the block to know the average.
  pos = 0.0f;
  posCount = 0;
  neg = 0.0f;
  negCount = 0;

  // There is always at least one warp executing this, and the number
  // of warps will be <= the number of lanes in a warp, so each warp
  // is guaranteed to get all values here.
  if (getLaneId() < blockDim.x / WARP_SIZE) {
    pos = blockPos[getLaneId()];
    posCount = blockPosCount[getLaneId()];
    neg = blockNeg[getLaneId()];
    negCount = blockNegCount[getLaneId()];
  }

  // Warp sum all the values
  pos = warpReduceSum(pos);
  posCount = warpReduceSum(posCount);
  neg = warpReduceSum(neg);
  negCount = warpReduceSum(negCount);

  // Now all threads in all warps have the final average.

  if (posCount > 0) {
    pos /= (float) posCount;
  }
  if (negCount > 0) {
    neg /= (float) negCount;
  }

  // Only one response per row
  if (threadIdx.x == 0) {
    avgPos[row] = pos;
    avgNeg[row] = neg;
  }

  // Quantize values
  int quantizedCol = 0;

  // Each warp reads up to 32 values from the row, quantizes them to a
  // single bit, and then we shuffle reduce to one unsigned int per
  // warp, which we write out.
  for (int warpCol = WARP_SIZE * getWarpId();
       warpCol < in.getSize(1);
       warpCol += blockDim.x) {
    int col = warpCol + getLaneId();

    unsigned on = 0;
    if (col < in.getSize(1)) {
      float val = in[row][col] + quantizationError[row][col];

      float error = 0.0f;

      if (val >= 0) {
        on = 1;
        error = val - pos;
      } else {
        error = val - neg;
      }

      // Each thread produces an error per entry.
      quantizationError[row][col] = error;
    }

    // Each lane will write a bit into the output if their value is >=
    // 0. Data is effectively then little endian with regards to input
    // order.
    const unsigned warpQuantized = __ballot(on);

    if (getLaneId() == 0) {
      out[row][warpCol / WARP_SIZE] = warpQuantized;
    }

    ++quantizedCol;
  }
}

__global__ void
dequantize(const DeviceTensor<unsigned, 2> in,
           const DeviceTensor<float, 1> avgPos,
           const DeviceTensor<float, 1> avgNeg,
           DeviceTensor<float, 2> out) {
  // This code only works if this is the case
  assert(warpSize == WARP_SIZE && WARP_SIZE == NUM_BITS);

  const int row = getRow();
  float pos = avgPos[row];
  float neg = avgNeg[row];

  // Each warp present will read a single quantized unsigned int, and
  // each thread will write out a single dequantized value
  for (int quantizedCol = getWarpId();
       quantizedCol < in.getSize(1);
       quantizedCol += blockDim.x / WARP_SIZE) {
    // All threads in the warp read the same quantized value
    const unsigned quantized = in[row][quantizedCol];

    // Each lane will read a bit from the input; data is effectively
    // little endian with regards to input order.
    const int bit = getBit(quantized, getLaneId());
    const float dequantized = bit ? pos : neg;

    // Of course, some of the bits may not correspond to real outputs
    const int dequantizedCol = quantizedCol * WARP_SIZE + getLaneId();
    if (dequantizedCol < out.getSize(1)) {
      out[row][dequantizedCol] = dequantized;
    }
  }
}

} // namespace

void
runQuantize1Bit(cudaStream_t stream,
                const DeviceTensor<float, 2>& in,
                DeviceTensor<float, 2>& out,
                DeviceTensor<float, 2>& quantizationError,
                DeviceTensor<float, 1>& avgPos,
                DeviceTensor<float, 1>& avgNeg) {
  // cutorch doesn't know about anything besides float tensors; create
  // one as unsigned
  DeviceTensor<unsigned, 2> outBit = out.cast<unsigned>();
  assert(outBit.getSize(0) == in.getSize(0));
  assert(ceil(in.getSize(1), NUM_BITS) == outBit.getSize(1));

  dim3 grid(in.getSize(0));

  // 4 seems to be a good sweet spot for number of warps to use, but
  // it doesn't make sense to have more warps than we have columns of
  // input
  int numWarps = min(4, ceil(in.getSize(1), WARP_SIZE));
  dim3 block(WARP_SIZE * numWarps);

  detail::quantize<<<grid, block, 0, stream>>>(
    in, outBit, quantizationError, avgPos, avgNeg);
}

void
runDequantize1Bit(cudaStream_t stream,
                  const DeviceTensor<float, 2>& in,
                  const DeviceTensor<float, 1>& avgPos,
                  const DeviceTensor<float, 1>& avgNeg,
                  DeviceTensor<float, 2>& out) {
  // cutorch doesn't know about anything besides float tensors; create
  // one as unsigned
  DeviceTensor<unsigned, 2> inBit = in.cast<unsigned>();
  assert(inBit.getSize(0) == out.getSize(0));
  assert(ceil(out.getSize(1), NUM_BITS) == inBit.getSize(1));

  dim3 grid(in.getSize(0));

  // 4 seems to be a good sweet spot for number of warps to use, but
  // it doesn't make sense to have more warps than we have columns of
  // input. The warps are based on the dequantized output (one thread
  // per value).
  int numWarps = min(4, ceil(out.getSize(1), WARP_SIZE));
  dim3 block(WARP_SIZE * numWarps);

  detail::dequantize<<<grid, block, 0, stream>>>(inBit, avgPos, avgNeg, out);
}

} } }

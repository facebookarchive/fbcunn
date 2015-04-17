#include <HalfPrec.h>
#include <gtest/gtest.h>
#include <common/math/Float16.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <math.h>

void cudaCheck(cudaError_t e) {
  auto toStr = [&] {
    return std::string(cudaGetErrorString(e));
  };
  if (e != cudaSuccess) {
    throw std::runtime_error("cuda failure: " + toStr());
  }
  e = cudaDeviceSynchronize();
  if (e != cudaSuccess) {
    throw std::runtime_error("cuda failure @ synchronize: " + toStr());
  }
}

template<typename T>
class CUDA {
public:
  explicit CUDA(size_t n)
  : n_(n) {
    cudaCheck(cudaMalloc(&vals_, n_ * sizeof(T)));
    cudaCheck(cudaMemset(vals_, 0, n_ * sizeof(T)));
  }

  CUDA(const T* base, size_t n) :
  n_(n) {
    cudaCheck(cudaMalloc(&vals_, n_ * sizeof(T)));
    cudaCheck(cudaMemcpy(vals_, base, n_ * sizeof(T), cudaMemcpyHostToDevice));
  }

  void toHost(T* base) const {
    cudaCheck(cudaMemcpy(base, vals_, n_ * sizeof(T), cudaMemcpyDeviceToHost));
  }

  size_t size() const {
    return n_;
  }

  ~CUDA() {
    cudaCheck(cudaFree(vals_));
  }

  T* data() {
    return vals_;
  }

private:
  T* vals_;
  size_t n_;
};

TEST(HalfPrec, cuda) {
  float hostFloats[] = {
    -1,
    -100,
    2.3,
    0.0,
    1.0,
    3867.2,
  };
  const auto N = sizeof(hostFloats) / sizeof(float);
  CUDA<float> devFloats(hostFloats, N);
  CUDA<half_t> devHalfs(N);

  halfprec_ToHalf(nullptr, devFloats.data(), devHalfs.data(), devFloats.size());
  cudaCheck(cudaDeviceSynchronize());

  {
    uint16_t cpuHalfs[N] = { 666 };
    facebook::math::Float16::encode(cpuHalfs, hostFloats, N);

    half_t convertedHalfs[N];
    devHalfs.toHost(convertedHalfs);
    for (int i = 0; i < N; i++) {
      // The CPU and GPU disagree by a digit sometimes because the GPU
      // is using a different rounding mode.
      EXPECT_NEAR(cpuHalfs[i], convertedHalfs[i], 1);
    }
  }

  CUDA<float> exploded(N);
  halfprec_ToFloat(nullptr, devHalfs.data(), exploded.data(), N);
  float postExpl[N];
  exploded.toHost(postExpl);
  for (int i = 0; i < N; i++) {
    auto thousandth = fabs(hostFloats[i] / 1000.0);
    EXPECT_NEAR(postExpl[i], hostFloats[i], thousandth);
  }
}

int halfSign(half_t h) {
  return (h & (1 << 15)) >> 15;
}

int halfExp(half_t h) {
  return (h >> 10) & 31;
}

int halfMant(half_t h) {
  return h & 1023;
}

TEST(HalfPrec, exhaustive) {
  const auto N = 1 << 16;

  half_t hostHalfs[N];
  float hostFloats[N];
  for (int i = 0; i < N; i++) {
    hostHalfs[i] = i;
  }
  facebook::math::Float16::decode(hostFloats, hostHalfs, N);

  CUDA<half_t> devHalfs(hostHalfs, N);
  CUDA<float> devFloats(N);
  float devOut[N];
  halfprec_ToFloat(nullptr, devHalfs.data(), devFloats.data(), N);
  devFloats.toHost(devOut);
  for (int i = 0; i < N; i++) {
    if (halfExp(i) == 0) continue; // subnormals
    if (halfExp(i) == 31) continue; // inf/nan
    if (hostFloats[i] != devOut[i]) {
      printf("failure: %d %x s/e/m %01x %03x %04x\n",
             i, i, halfSign(i), halfExp(i), halfMant(i));
      EXPECT_EQ(hostFloats[i], devOut[i]);
    }
  }
}

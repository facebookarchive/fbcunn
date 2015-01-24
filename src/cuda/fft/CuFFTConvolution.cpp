// Copyright 2004-present Facebook. All Rights Reserved.

#include "fft/CuFFTConvolution.h"
#include "fft/CuFFTConvolution.cuh"

#include "cuda/DeviceTensor.cuh"
#include "DeviceTensorUtils.h"
#include "THCTensor.h"
#include "CuBLASWrapper.h"
#include "fft/CuFFTWrapper.cuh"
#include "fft/Utils.cuh"
#include "fft/Utils.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <folly/Bits.h>
#include <folly/Hash.h>
#include <folly/Memory.h>
#include <folly/ScopeGuard.h>
#include <glog/logging.h>
#include <unordered_map>
#include <unordered_set>

using namespace facebook::deeplearning::torch;

using namespace std;

namespace facebook { namespace deeplearning { namespace torch {

typedef CuFFTConvolution::CudaRealTensor CudaRealTensor;
typedef CuFFTConvolution::CudaComplexTensor CudaComplexTensor;

THParams::THParams(THCudaTensor* in,
                   THCudaTensor* wei,
                   THCudaTensor* out,
                   THCudaTensor* b,
                   float sc) :
    input(in),
    weight(wei),
    output(out),
    bias(b),
    scale(sc) {}

void THParams::free() {
  THCudaTensor_free(NULL, input);
  input = nullptr;
  THCudaTensor_free(NULL, weight);
  weight = nullptr;
  THCudaTensor_free(NULL, output);
  output = nullptr;
  THCudaTensor_free(NULL, bias);
  bias = nullptr;
}

ProblemSizes::ProblemSizes(const THParams& params, ConvolutionPass p)  {
  pass = p;
  if (pass.pass == ConvolutionPass::kUpdateOutput) {
    // Output is sometimes not allocated here, recover its size from the
    // input sizes
    batchSize = THCudaTensor_size(NULL, params.input, 0);
    filterSize = THCudaTensor_size(NULL, params.weight, 0);
    planeSize = THCudaTensor_size(NULL, params.input, 1);
    inputSizeRow = THCudaTensor_size(NULL, params.input, 2);
    inputSizeCol = THCudaTensor_size(NULL, params.input, 3);
    weightSizeRow = THCudaTensor_size(NULL, params.weight, 2);
    weightSizeCol = THCudaTensor_size(NULL, params.weight, 3);
    outputSizeRow = (inputSizeRow - weightSizeRow) + 1;
    outputSizeCol = (inputSizeCol - weightSizeCol) + 1;
  } else if (pass.pass == ConvolutionPass::kUpdateGradInput) {
    // Input is sometimes not allocated here, recover its size from the
    // output sizes
    batchSize = THCudaTensor_size(NULL, params.output, 0);
    filterSize = THCudaTensor_size(NULL, params.output, 1);
    planeSize = THCudaTensor_size(NULL, params.weight, 1);
    outputSizeRow = THCudaTensor_size(NULL, params.output, 2);
    outputSizeCol = THCudaTensor_size(NULL, params.output, 3);
    weightSizeRow = THCudaTensor_size(NULL, params.weight, 2);
    weightSizeCol = THCudaTensor_size(NULL, params.weight, 3);
    inputSizeRow = (outputSizeRow - 1) + weightSizeRow;
    inputSizeCol = (outputSizeCol - 1) + weightSizeCol;
  } else {
    // All 3 tensors are allocated here
    CHECK(pass.pass == ConvolutionPass::kAccGradParameters);
    batchSize = THCudaTensor_size(NULL, params.input, 0);
    filterSize = THCudaTensor_size(NULL, params.weight, 0);
    planeSize = THCudaTensor_size(NULL, params.input, 1);
    inputSizeRow = THCudaTensor_size(NULL, params.input, 2);
    inputSizeCol = THCudaTensor_size(NULL, params.input, 3);
    weightSizeRow = THCudaTensor_size(NULL, params.weight, 2);
    weightSizeCol = THCudaTensor_size(NULL, params.weight, 3);
    outputSizeRow = THCudaTensor_size(NULL, params.output, 2);
    outputSizeCol = THCudaTensor_size(NULL, params.output, 3);
  }
  expandedSizeRow = std::max(std::max(inputSizeRow, outputSizeRow),
                             weightSizeRow);
  expandedSizeCol = std::max(std::max(inputSizeCol, outputSizeCol),
                             weightSizeCol);
}

// This is used for autotuning, given a problem size fit
THParams ProblemSizes::makeTensors() const {
  auto input = makeTHCudaTensorFull({
      batchSize, planeSize, inputSizeRow, inputSizeCol});
  auto filters = makeTHCudaTensorFull({
      filterSize, planeSize, weightSizeRow, weightSizeCol});
  auto output = makeTHCudaTensorFull({
      batchSize, filterSize, outputSizeRow, outputSizeCol});
  auto bias = makeTHCudaTensorFull({filterSize});
  return THParams(input.release(),
                  filters.release(),
                  output.release(),
                  bias.release());
}

namespace {
bool containsOtherPrimes(long i, const std::vector<long>& primeFactors) {
  DCHECK_LE(1, i);
  for (auto f : primeFactors) {
    while (i % f == 0) {
      i /= f;
    }
  }
  return i > 1;
}

// Returns a vector of powers of "primeFactors" greater than i for FFT size
// candidates.
// In the particular case where i is already a power of 2, just returns i.
// @param primeFactors should contain unique prime factors. This method will
// sort the factors for its internal use but will otherwise make no checks on
// whether factors are prime or unique.
vector<long> makePowersGreaterThan(long i, std::vector<long> primeFactors) {
  DCHECK_LT(0, i);
  auto msb = folly::findLastSet(i);
  auto msbVal = 1 << (msb - 1);
  DCHECK_GE(i, msbVal);
  DCHECK_LE(i, msbVal << 1);
  if (msbVal == i) {
    return vector<long>({i});
  }

  std::sort(primeFactors.begin(), primeFactors.end());

  // If not a power of 2, produce all multiples of 2.3.5.7
  // in range[i, (msbVal << 1)]
  vector<long> result({i});
  unordered_set<long> candidates;
  for (auto p : primeFactors) {
    for (auto f = std::max(1L, i / primeFactors.back()); f < i; ++f) {
      if (containsOtherPrimes(f, primeFactors)) {
        continue;
      }
      auto val = f * p;
      if (val <= i || val >= (msbVal << 1)) {
        continue;
      }
      if (candidates.count(val) > 0) {
        continue;
      }
      candidates.insert(val);
      result.push_back(val);
    }
  }
  result.push_back(msbVal << 1);

  return result;
}
}

std::vector<long> ProblemSizes::sizes() const {
  return std::vector<long>({rows(), cols()});
}

std::vector<CuFFTStrategy> CuFFTStrategy::makeStrategies() const {
  std::vector<CuFFTStrategy> res({*this});

  {
    // Add batch vs many mode
    auto copy = res;
    res.clear();
    for (auto b : {true, false}) {
      for (auto strat : copy) {
        res.push_back(strat.withBatch(b));
      }
    }
  }

  {
    // Add rows
    auto copy = res;
    res.clear();
    for (auto strat : copy) {
      for (auto row : makePowersGreaterThan(strat.sizes.rows(),
                                            vector<long>({2, 3, 5, 7}))) {
        auto s = strat;
        s.sizes.withExpandedSizeRow(row);
        res.push_back(s);
      }
    }
  }

  {
    // Add cols
    auto copy = res;
    res.clear();
    for (auto strat : copy) {
      for (auto col : makePowersGreaterThan(strat.sizes.cols(),
                                             vector<long>({2, 3, 5, 7}))) {
        auto s = strat;
        s.sizes.withExpandedSizeCol(col);
        res.push_back(s);
      }
    }
  }

  return res;
}

std::ostream& operator<<(std::ostream& os, const CuFFTStrategy& s) {
  if (s.batch) {
    os << "(Batch) ";
  } else {
    os << "(Many) ";
  }
  os << s.sizes;
  return os;
}

std::ostream& operator<<(std::ostream& os, const ProblemSizes& pbs) {
  os << "(b x p x f) = " << pbs.batchSize << "x" <<
     pbs.planeSize << "x" << pbs.filterSize << " ";
  os << "(input rows x cols) = " << pbs.inputSizeRow << "x" <<
    pbs.inputSizeCol << " ";
  os << "(filter rows x cols) = " << pbs.weightSizeRow << "x" <<
    pbs.weightSizeCol << " ";
  os << "(common rows x cols) = " << pbs.rows() << "x" << pbs.cols() << " ";
  return os;
}


CuFFTConvolution&
CuFFTConvolution::withInputAndBuffers(
    THCudaTensor* inputTH,
    THCudaTensor* inputComplexTH,
    THCudaTensor* inputComplexTrTH,
    cufftHandle* plan,
    cufftHandle* planInverse) {
  if (convPass_.pass == ConvolutionPass::kUpdateOutput) {
    B_ = torchToDeviceTensor<float, 4>(inputTH);
    BComplex_ = torchToDeviceTensor<float, 5>(inputComplexTH);
    BComplexT_ = torchToDeviceTensor<float, 5>(inputComplexTrTH);
    fftPlanB_ = plan;
    fftPlanInverseB_ = planInverse;
  } else if (convPass_.pass == ConvolutionPass::kAccGradParameters) {
    A_ = torchToDeviceTensor<float, 4>(inputTH);
    AComplex_ = torchToDeviceTensor<float, 5>(inputComplexTH);
    AComplexT_ = torchToDeviceTensor<float, 5>(inputComplexTrTH);
    fftPlanA_ = plan;
    fftPlanInverseA_ = planInverse;
  } else {
    DCHECK_EQ(ConvolutionPass::kUpdateGradInput, convPass_.pass);
    C_ = torchToDeviceTensor<float, 4>(inputTH);
    CComplex_ = torchToDeviceTensor<float, 5>(inputComplexTH);
    CComplexT_ = torchToDeviceTensor<float, 5>(inputComplexTrTH);
    fftPlanC_ = plan;
  }
  return *this;
}

CuFFTConvolution&
CuFFTConvolution::withOutputAndBuffers(
    THCudaTensor* outputTH,
    THCudaTensor* outputComplexTH,
    THCudaTensor* outputComplexTrTH,
    cufftHandle* plan,
    cufftHandle* planInverse) {
  if (convPass_.pass == ConvolutionPass::kUpdateOutput) {
    C_ = torchToDeviceTensor<float, 4>(outputTH);
    CComplex_ = torchToDeviceTensor<float, 5>(outputComplexTH);
    CComplexT_ = torchToDeviceTensor<float, 5>(outputComplexTrTH);
    fftPlanC_ = plan;
  } else if (convPass_.pass == ConvolutionPass::kAccGradParameters) {
    B_ = torchToDeviceTensor<float, 4>(outputTH);
    BComplex_ = torchToDeviceTensor<float, 5>(outputComplexTH);
    BComplexT_ = torchToDeviceTensor<float, 5>(outputComplexTrTH);
    fftPlanB_ = plan;
    fftPlanInverseB_ = planInverse;
  } else {
    DCHECK_EQ(ConvolutionPass::kUpdateGradInput, convPass_.pass);
    B_ = torchToDeviceTensor<float, 4>(outputTH);
    BComplex_ = torchToDeviceTensor<float, 5>(outputComplexTH);
    BComplexT_ = torchToDeviceTensor<float, 5>(outputComplexTrTH);
    fftPlanB_ = plan;
    fftPlanInverseB_ = planInverse;
  }
  return *this;
}

CuFFTConvolution&
CuFFTConvolution::withFiltersAndBuffers(
    THCudaTensor* filtersTH,
    THCudaTensor* filtersComplexTH,
    THCudaTensor* filtersComplexTrTH,
    cufftHandle* plan,
    cufftHandle* planInverse) {
  if (convPass_.pass == ConvolutionPass::kUpdateOutput) {
    A_ = torchToDeviceTensor<float, 4>(filtersTH);
    AComplex_ = torchToDeviceTensor<float, 5>(filtersComplexTH);
    AComplexT_ = torchToDeviceTensor<float, 5>(filtersComplexTrTH);
    fftPlanA_ = plan;
    fftPlanInverseA_ = planInverse;
  } else if (convPass_.pass == ConvolutionPass::kAccGradParameters) {
    C_ = torchToDeviceTensor<float, 4>(filtersTH);
    CComplex_ = torchToDeviceTensor<float, 5>(filtersComplexTH);
    CComplexT_ = torchToDeviceTensor<float, 5>(filtersComplexTrTH);
    fftPlanC_ = plan;
  } else {
    DCHECK_EQ(ConvolutionPass::kUpdateGradInput, convPass_.pass);
    A_ = torchToDeviceTensor<float, 4>(filtersTH);
    AComplex_ = torchToDeviceTensor<float, 5>(filtersComplexTH);
    AComplexT_ = torchToDeviceTensor<float, 5>(filtersComplexTrTH);
    fftPlanA_ = plan;
    fftPlanInverseA_ = planInverse;
  }
  return *this;
}


namespace {
class CuFFTBuffers {
 public:
  int currentDevice() {
    int currentDevice;
    THCudaCheck(cudaGetDevice(&currentDevice));
    return currentDevice;
  }

  static CuFFTBuffers& singleton() {
    static CuFFTBuffers bufs;
    return bufs;
  }

  enum struct MatrixName : int {
    A = 1,
      B,
      C
  };

  // Matrix name is 'A', 'B' or 'C', make it into a type if more needed
  template <typename ElemPtr>
  ElemPtr* devNamesToAlloc(MatrixName name,
                           ElemPtr first,
                           ElemPtr last,
                           cudaStream_t stream,
                           cudaEvent_t event) {
    DCHECK_LE((int)MatrixName::A, (int)name);
    DCHECK_GE((int)MatrixName::C, (int)name);

    auto kSize = (last - first + 1) * sizeof(void*);
    DCHECK_LT(0, kSize);

    DeviceMatrixName key(currentDevice(), (int)name);
    // If we need to resize just delete and realloc, this is monotonic in the
    // size and only happens a negligible number of times.
    bool alloc = true;
    if (devNamesToAlloc_.count(key) > 0) {
      DCHECK_EQ(1, devNamesToAlloc_.count(key)); // collisions not allowed
      if (devNamesToAlloc_[key].size < kSize) {
        cudaFree(devNamesToAlloc_[key].devicePtr);
        devNamesToAlloc_.erase(key);
      } else {
        // Enough space, just reuse
        alloc = false;
      }
    }

    if (alloc) {
      // If first time or realloc, increase size only
      devNamesToAlloc_[key].size = kSize;
      devNamesToAlloc_[key].devicePtr = nullptr;
      CHECK_EQ(cudaSuccess,
               cudaMalloc(&devNamesToAlloc_[key].devicePtr,
                          kSize * sizeof(void*)));
      DCHECK(devNamesToAlloc_[key].devicePtr);
    }

    // Enough size must be allocated on device
    DCHECK_LE(kSize, devNamesToAlloc_[key].size);
    DCHECK(devNamesToAlloc_[key].devicePtr);
    if (stream) {
      DCHECK(event);
      CHECK_EQ(cudaSuccess,
               cudaMemcpyAsync(devNamesToAlloc_[key].devicePtr,
                               // std::vector storage is contiguous in memory
                               first,
                               kSize,
                               cudaMemcpyHostToDevice,
                               stream
                              ));

      THCudaCheck(cudaEventRecord(event, stream));
    } else {
      // No stream, pay the price
      CHECK_EQ(cudaSuccess,
               cudaMemcpy(devNamesToAlloc_[key].devicePtr,
                          // std::vector storage is contiguous in memory
                          first,
                          kSize,
                          cudaMemcpyHostToDevice
                         ));
    }

    return (ElemPtr*)(devNamesToAlloc_[key].devicePtr);
  }

  void reset() {
    for (auto& kvp : devNamesToAlloc_) {
      cudaFree(kvp.second.devicePtr);
    }
    devNamesToAlloc_.clear();
  }
 private:
  CuFFTBuffers(const CuFFTBuffers&) = delete;
  CuFFTBuffers& operator=(CuFFTBuffers&) = delete;

  CuFFTBuffers() {}

  ~CuFFTBuffers() {
    reset();
  }

  typedef int cudaDevice_t;
  typedef int matrixName_t;

  struct DevicePointerSize {
    size_t size;
    void* devicePtr;
  };
  typedef std::tuple<cudaDevice_t, matrixName_t> DeviceMatrixName;
  std::unordered_map<DeviceMatrixName, DevicePointerSize> devNamesToAlloc_;
};
} // namespace

void CuFFTConvolution::reset() {
  CuFFTBuffers::singleton().reset();
}


void CuFFTConvolution::allToAllWait() {
  // Register every stream to its event and make stream 0 wait for them
  // Don't screw up asynchrony
  // If no events from above, then just cudaDeviceSynchronize
  if (!cudaStreams_.empty() && !cudaEvents_.empty()) {
    kWaitsOnAll(0);
    allWaitOnK(0);
  } else {
    THCudaCheck(cudaDeviceSynchronize());
  }
  THCudaCheck(cudaGetLastError());
}

void CuFFTConvolution::kWaitsOnL(int k, int l) {
  // Register stream l to its event and make stream k wait for it
  // Don't screw up asynchrony
  // If no events from above, then just cudaDeviceSynchronize
  if (cudaStreams_.size() > std::max(k, l) &&
      cudaEvents_.size() > std::max(k, l)) {
    DCHECK_EQ(cudaEvents_.size(), cudaStreams_.size());
    CHECK_EQ(cudaSuccess,
             cudaEventRecord(getEvent(l), getStream(l)));
    CHECK_EQ(cudaSuccess,
             cudaStreamWaitEvent(getStream(k), getEvent(l), 0));
  } else {
    THCudaCheck(cudaDeviceSynchronize());
  }
  THCudaCheck(cudaGetLastError());
}

void CuFFTConvolution::kWaitsOnAll(int k) {
  // Register every stream to its event and make stream k wait for them
  // Don't screw up asynchrony
  // If no events from above, then just cudaDeviceSynchronize
  if (!cudaStreams_.empty() && !cudaEvents_.empty()) {
    DCHECK_EQ(cudaEvents_.size(), cudaStreams_.size());
    for (int i = 0; i < cudaStreams_.size(); ++i) {
      CHECK_EQ(cudaSuccess,
               cudaEventRecord(getEvent(i), getStream(i)));
      CHECK_EQ(cudaSuccess,
               cudaStreamWaitEvent(getStream(k), getEvent(i), 0));
    }
  } else {
    THCudaCheck(cudaDeviceSynchronize());
  }
  THCudaCheck(cudaGetLastError());
}

void CuFFTConvolution::allWaitOnK(int k) {
  // Register every stream to its event and make them wait for stream k
  // Don't screw up asynchrony
  // If no events from above, then just cudaDeviceSynchronize
  if (!cudaStreams_.empty() && !cudaEvents_.empty()) {
    DCHECK_EQ(cudaEvents_.size(), cudaStreams_.size());
    CHECK_EQ(cudaSuccess,
             cudaEventRecord(getEvent(k), getStream(k)));
    for (int i = 0; i < cudaStreams_.size(); ++i) {
      CHECK_EQ(cudaSuccess,
               cudaStreamWaitEvent(getStream(i), getEvent(k), 0));
    }
  } else {
    THCudaCheck(cudaDeviceSynchronize());
  }
  THCudaCheck(cudaGetLastError());
}


void CuFFTConvolution::CuFFTConvolutionMxMBatch(
  const std::vector<cublasHandle_t>& handles) {
  CHECK_LE(1, handles.size());
  auto handle = handles[0];
  auto& A_ = AComplexT_;
  auto& B_ = BComplexT_;
  auto& C_ = CComplexT_;

  // All have same row / cols in Fourier domain, at this point, transposition
  // must have occurred
  const int kNumRows = A_.getSize(0);
  const int kNumCols = A_.getSize(1);
  // Complex storage already exploits hermitian symmetry along cols
  const auto kNumHermitianCols = kNumCols;

  DCHECK_EQ(kNumRows, B_.getSize(0));
  DCHECK_EQ(kNumCols, B_.getSize(1));
  DCHECK_EQ(kNumRows, C_.getSize(0));
  DCHECK_EQ(kNumCols, C_.getSize(1));

  std::vector<const cuFloatComplex*> APtrVec;
  std::vector<const cuFloatComplex*> BPtrVec;
  std::vector<cuFloatComplex*> CPtrVec;

  for (int row = 0; row < kNumRows; ++row) {
    for (int col = 0; col < kNumHermitianCols; ++col) {
      auto APtr = A_[0].dataAs<cuFloatComplex>();
      APtr += (row * kNumCols + col) * A_.getSize(2) * A_.getSize(3);
      APtrVec.push_back(APtr);

      auto BPtr = B_[0].dataAs<cuFloatComplex>();
      BPtr += (row * kNumCols + col) * B_.getSize(2) * B_.getSize(3);
      BPtrVec.push_back(BPtr);

      auto CPtr = C_[0].dataAs<cuFloatComplex>();
      CPtr += (row * kNumCols + col) * C_.getSize(2) * C_.getSize(3);
      CPtrVec.push_back(CPtr);
    }
  }

  auto& buf = CuFFTBuffers::singleton();

  // Even seemingly insignificant thrust host->device ptr copies can cause
  // painful synchronizations
  auto APtrCtrDevice = (const cuComplex**)
    buf.devNamesToAlloc(CuFFTBuffers::MatrixName::A, &APtrVec.front(),
                        &APtrVec.back(), getStream(0), getEvent(0));
  auto BPtrCtrDevice = (const cuComplex**)
    buf.devNamesToAlloc(CuFFTBuffers::MatrixName::B, &BPtrVec.front(),
                        &BPtrVec.back(), getStream(1), getEvent(1));
  auto CPtrCtrDevice = (cuComplex**)
    buf.devNamesToAlloc(CuFFTBuffers::MatrixName::C, &CPtrVec.front(),
                        &CPtrVec.back(), getStream(2), getEvent(2));

  DCHECK_EQ(kNumRows * kNumHermitianCols, APtrVec.size());

  // Synchronize streams
  if (!cudaStreams_.empty() && !cudaEvents_.empty()) {
    for (int i = 0; i < 3; ++i) {
      CHECK_EQ(cudaSuccess,
               cudaStreamWaitEvent(getStream(0), getEvent(i), 0));
    }
    // And run MxM on stream 0
    cublasSetStream(handle, getStream(0));
  }

  auto zero = make_cuComplex(0.0f, 0.0f);
  auto res =
    cublasCgemmBatched(handle,
                       transA_,
                       transB_,
                       C_.getSize(3),
                       C_.getSize(2),
                       mmReductionSize_,
                       &norm_,
                       &APtrCtrDevice[0],
                       fastestVaryingSizeA_,
                       &BPtrCtrDevice[0],
                       fastestVaryingSizeB_,
                       &zero,
                       &CPtrCtrDevice[0],
                       C_.getSize(3),
                       kNumRows * kNumHermitianCols
                      );
  DCHECK_EQ(CUBLAS_STATUS_SUCCESS, res);
}

void CuFFTConvolution::CuFFTConvolutionMxMMany() {
  auto& A_ = AComplexT_;
  auto& B_ = BComplexT_;
  auto& C_ = CComplexT_;

  // All have same row / cols in Fourier domain, at this point, permutation
  // has occurred
  const int kNumRows = A_.getSize(0);
  const int kNumCols = A_.getSize(1);
  // Complex storage already exploits hermitian symmetry along cols
  const auto kNumHermitianCols = kNumCols;

  DCHECK_EQ(kNumRows, B_.getSize(0));
  DCHECK_EQ(kNumCols, B_.getSize(1));
  DCHECK_EQ(kNumRows, C_.getSize(0));
  DCHECK_EQ(kNumCols, C_.getSize(1));

  // If you want this version you need as many streams as you have handles
  CHECK(!cudaStreams_.empty());
  CHECK_EQ(cudaStreams_.size(), cublasHandles_.size());
  {
    const auto zero = make_cuComplex(0.0f, 0.0f);

    for (int i = 0; i < cudaStreams_.size(); ++i) {
      {
        // Prepare async cgemm calls
        auto res =
          cublasSetStream(cublasHandles_[i], getStream(i));
        DCHECK_EQ(CUBLAS_STATUS_SUCCESS, res);
      }
    }
    for (int row = 0; row < kNumRows; ++row) {
      for (int col = 0; col < kNumHermitianCols; ++col) {
        auto ind = (row * kNumHermitianCols + col) % cudaStreams_.size();
        auto APtr = A_[0].dataAs<cuFloatComplex>();
        APtr += (row * kNumCols + col) * A_.getSize(2) * A_.getSize(3);

        auto BPtr = B_[0].dataAs<cuFloatComplex>();
        BPtr += (row * kNumCols + col) * B_.getSize(2) * B_.getSize(3);

        auto CPtr = C_[0].dataAs<cuFloatComplex>();
        CPtr += (row * kNumCols + col) * C_.getSize(2) * C_.getSize(3);

        auto res = cublasCgemm(cublasHandles_[ind],
                               transA_,
                               transB_,
                               C_.getSize(3),
                               C_.getSize(2),
                               mmReductionSize_,
                               &norm_,
                               APtr,
                               fastestVaryingSizeA_,
                               BPtr,
                               fastestVaryingSizeB_,
                               &zero,
                               CPtr,
                               C_.getSize(3)
                              );
        DCHECK_EQ(CUBLAS_STATUS_SUCCESS, res);
      }
    }
  }
}

void CuFFTConvolution::CuFFTConvolutionMxM() {
  if (strategy_->batch) {
    // Create local handle if not passed during configuration
    // If no handle from above this will implicitly create a synchronization on
    // destruction.
    auto localHandles = cublasHandles_;
    if (localHandles.empty()) {
      if (cudaStreams_.empty()) {
        localHandles.push_back(cublasHandle_t());
        cublasCreate(&(localHandles.back()));
      } else {
        for (int i = 0; i < cudaStreams_.size(); ++i) {
          localHandles.push_back(cublasHandle_t());
          cublasCreate(&(localHandles.back()));
        }
      }
    } else {
      // Need one handle per stream
      CHECK_EQ(cudaStreams_.size(), cublasHandles_.size());
      localHandles = cublasHandles_;
    }
    SCOPE_EXIT {
      if (cublasHandles_.empty()) {
        for (auto& h : localHandles) {
          cublasDestroy(h);
        }
      }
    };
    CuFFTConvolutionMxMBatch(localHandles);
  } else {
    // If you want this version you must set up streams properly
    CuFFTConvolutionMxMMany();
  }
}

void CuFFTConvolution::run() {
  DCHECK(nullptr != A_.data());
  DCHECK(nullptr != B_.data());
  DCHECK(nullptr != C_.data());
  DCHECK(nullptr != AComplex_.data());
  DCHECK(nullptr != BComplex_.data());
  DCHECK(nullptr != CComplex_.data());
  DCHECK(nullptr != AComplexT_.data());
  DCHECK(nullptr != BComplexT_.data());
  DCHECK(nullptr != CComplexT_.data());

  // 1. Setup MxM properties from pass
  if (convPass_.pass == ConvolutionPass::kUpdateOutput) {
    mmReductionSize_   = A_.getSize(1);   // numInputPlanes
    fastestVaryingSizeA_ = A_.getSize(1); // numInputPlanes
    fastestVaryingSizeB_ = B_.getSize(1); // numInputPlanes
    transA_ = CUBLAS_OP_C;
    transB_ = CUBLAS_OP_N;
  } else if (convPass_.pass == ConvolutionPass::kAccGradParameters) {
    mmReductionSize_   = A_.getSize(0);   // numBatches
    fastestVaryingSizeA_ = A_.getSize(1); // numInputPlanes
    fastestVaryingSizeB_ = B_.getSize(1); // numFilters
    transA_ = CUBLAS_OP_N;
    transB_ = CUBLAS_OP_C;
  } else {
    DCHECK_EQ(ConvolutionPass::kUpdateGradInput, convPass_.pass);
    mmReductionSize_   = A_.getSize(0);   // numFilters
    fastestVaryingSizeA_ = A_.getSize(1); // numInputPlanes
    fastestVaryingSizeB_ = B_.getSize(1); // numFilters
    transA_ = CUBLAS_OP_N;
    transB_ = CUBLAS_OP_N;
  }

  if (!strategy_) {
    strategy_ = &CuFFTStrategy::defaultStrategy();
  }

  DCHECK_EQ(A_.getSize(2), B_.getSize(2));
  DCHECK_EQ(A_.getSize(3), B_.getSize(3));
  DCHECK_EQ(C_.getSize(2), B_.getSize(2));
  DCHECK_EQ(C_.getSize(3), B_.getSize(3));

  norm_ = make_cuComplex(scale_ / (A_.getSize(2) * A_.getSize(3)), 0.0f);

  cudaStream_t s0 = getStream(0);
  cudaStream_t s1 = getStream(1);
  auto handle0 = getCircular(cublasHandles_, 0);
  // FFT followed by transpose in same stream
  fft2d<2>(A_, AComplex_, FFTParameters().forward(), fftPlanA_, s0);
  // Transpose A_ (? ? y x) -> (y x ? ?) (row-major formulation)
  transposeAsComplex(AComplex_, AComplexT_, 2, handle0, s0);

  auto handle1 = getCircular(cublasHandles_, 1);
  // FFT followed by transpose in same stream
  fft2d<2>(B_, BComplex_, FFTParameters().forward(), fftPlanB_, s1);
  // Transpose A_ (? ? y x) -> (y x ? ?) (row-major formulation)
  transposeAsComplex(BComplex_, BComplexT_, 2, handle1, s1);

  // Here, both CComplex_ and CComplexT_ contain garbage that we will
  // overwrite and that we preemptively size to (y x ? ?)..
  // This is just resizing, not actual transposition since we don't need to
  // move garbage data anywhere.
  std::vector<int> perm({2, 3, 0, 1, 4});
  // This is a shallow permutation at the CudaTensor level, no data is moved.
  CComplex_.permuteDims(perm);
  CComplexT_.permuteDims(perm);

  if (!strategy_->batch) {
    allWaitOnK(0);
    allWaitOnK(1);
  } else {
    kWaitsOnL(0, 1);
  }

  // Run MxM
  CuFFTConvolutionMxM();
  if (!strategy_->batch) {
    kWaitsOnAll(0);
  }

  cudaStream_t s = getStream(0);
  auto handle = getCircular(cublasHandles_, 0);
  // Transpose followed by IFFT in same stream s0 as the MxM
  // Transpose input (y x ? ?) -> (? ? y x) (row-major formulation)
  transposeAsComplex(CComplexT_, CComplex_, 2, handle, s);
  fft2d<2>(C_, CComplex_, FFTParameters().inverse().normalize(false),
           fftPlanC_, s);

  // Everything is transitively synchronized on stream0 at this point
}

} } } // namespace

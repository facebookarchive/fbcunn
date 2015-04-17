// Copyright 2004-present Facebook. All Rights Reserved.

#include "CuFFTConvolution.cuh"

#include "THCTensor.h"
#include "cuda/DeviceTensor.cuh"
#include "CuBLASWrapper.h"
#include "DeviceTensorUtils.h"
#include "MM.h"
#include "CuFFTStrategy.h"
#include "CuFFTWrapper.cuh"
#include "FBFFTHost.h"
#include "Utils.cuh"
#include "Utils.h"

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

THParams::THParams(THCState* s,
                   THCudaTensor* in,
                   THCudaTensor* wei,
                   THCudaTensor* out,
                   THCudaTensor* b,
                   float sc,
                   THBuffers bufs) :
    state(s),
    input(in),
    weight(wei),
    output(out),
    bias(b),
    scale(sc),
    buffers(bufs) {}

void THParams::free() {
  THCudaTensor_free(state, input);
  input = nullptr;
  THCudaTensor_free(state, weight);
  weight = nullptr;
  THCudaTensor_free(state, output);
  output = nullptr;
  THCudaTensor_free(state, bias);
  bias = nullptr;
  buffers = THBuffers(); // Managed in lua land
}

ProblemSizes::ProblemSizes(const THParams& params, ConvolutionPass p)  {
  pass = p;
  if (pass.pass == ConvolutionPass::kUpdateOutput) {
    // Output is sometimes not allocated here, recover its size from the
    // input sizes
    batchSize = THCudaTensor_size(params.state, params.input, 0);
    filterSize = THCudaTensor_size(params.state, params.weight, 0);
    planeSize = THCudaTensor_size(params.state, params.input, 1);
    inputSizeRow = THCudaTensor_size(params.state, params.input, 2);
    inputSizeCol = THCudaTensor_size(params.state, params.input, 3);
    weightSizeRow = THCudaTensor_size(params.state, params.weight, 2);
    weightSizeCol = THCudaTensor_size(params.state, params.weight, 3);
    outputSizeRow = (inputSizeRow - weightSizeRow) + 1;
    outputSizeCol = (inputSizeCol - weightSizeCol) + 1;
  } else if (pass.pass == ConvolutionPass::kUpdateGradInput) {
    // Input is sometimes not allocated here, recover its size from the
    // output sizes
    batchSize = THCudaTensor_size(params.state, params.output, 0);
    filterSize = THCudaTensor_size(params.state, params.output, 1);
    planeSize = THCudaTensor_size(params.state, params.weight, 1);
    outputSizeRow = THCudaTensor_size(params.state, params.output, 2);
    outputSizeCol = THCudaTensor_size(params.state, params.output, 3);
    weightSizeRow = THCudaTensor_size(params.state, params.weight, 2);
    weightSizeCol = THCudaTensor_size(params.state, params.weight, 3);
    inputSizeRow = (outputSizeRow - 1) + weightSizeRow;
    inputSizeCol = (outputSizeCol - 1) + weightSizeCol;
  } else {
    // All 3 tensors are allocated here
    CHECK(pass.pass == ConvolutionPass::kAccGradParameters);
    batchSize = THCudaTensor_size(params.state, params.input, 0);
    filterSize = THCudaTensor_size(params.state, params.weight, 0);
    planeSize = THCudaTensor_size(params.state, params.input, 1);
    inputSizeRow = THCudaTensor_size(params.state, params.input, 2);
    inputSizeCol = THCudaTensor_size(params.state, params.input, 3);
    weightSizeRow = THCudaTensor_size(params.state, params.weight, 2);
    weightSizeCol = THCudaTensor_size(params.state, params.weight, 3);
    outputSizeRow = THCudaTensor_size(params.state, params.output, 2);
    outputSizeCol = THCudaTensor_size(params.state, params.output, 3);
  }
  expandedSizeRow = std::max(std::max(inputSizeRow, outputSizeRow),
                             weightSizeRow);
  expandedSizeCol = std::max(std::max(inputSizeCol, outputSizeCol),
                             weightSizeCol);
  buffers = params.buffers;
}

// This is used for autotuning, given a problem size fit
THParams ProblemSizes::makeTensors(THCState* state) const {
  auto input = makeTHCudaTensorFull(
    state,
    {batchSize, planeSize, inputSizeRow, inputSizeCol});
  auto filters = makeTHCudaTensorFull(
    state,
    {filterSize, planeSize, weightSizeRow, weightSizeCol});
  auto output = makeTHCudaTensorFull(
    state,
    {batchSize, filterSize, outputSizeRow, outputSizeCol});
  auto bias = makeTHCudaTensorFull(state, {filterSize});
  return THParams(state,
                  input.release(),
                  filters.release(),
                  output.release(),
                  bias.release(),
                  0.0f,
                  buffers);
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
  // FBFFT does not support sizes < 2, just force size of 2 instead
  // TODO (#4948477): Catch this earlier and avoid FFT altogether if
  // convolution is really of size 1.
  if (i == 1) {i = 2;}
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
  vector<long> result({msbVal << 1, i});
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
    for (auto b : {
        CuFFTStrategy::MMVersion::batchMM,
          CuFFTStrategy::MMVersion::manyMM,
          CuFFTStrategy::MMVersion::fbTransposeMM }) {
      for (auto strat : copy) {
        res.push_back(strat.withMM(b));
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
        if (s.maxDataElements() <= CuFFTStrategy::kMaxElements) {
          res.push_back(s);
        }
      }
    }
  }

  {
    for (auto b : {
        CuFFTStrategy::MMVersion::batchMM,
        CuFFTStrategy::MMVersion::manyMM,
        CuFFTStrategy::MMVersion::fbTransposeMM
          }) {
      // Add FBFFT variants
      CuFFTStrategy s(*this);
      // TODO (#4948477): No need to take the max, just take the msb for each
      // size -> FBFFT needs to support rectangular patches first
      auto maxSquareSize = std::max(
        makePowersGreaterThan(s.sizes.rows(),
                              vector<long>({2}))[0],
        makePowersGreaterThan(s.sizes.cols(),
                              vector<long>({2}))[0]);

      s.sizes.withExpandedSizeRow(maxSquareSize)
        .withExpandedSizeCol(maxSquareSize);

      res.push_back(s.withMM(b)
                    .withFFT(FFTParameters::FFTVersion::fbfft));
    }
  }

  return res;
}

std::ostream& operator<<(std::ostream& os, const CuFFTStrategy& s) {
  if (s.batchmm()) {
    os << "(Batch ";
  } else if (s.manymm()) {
    os << "(Many ";
  } else if (s.fbmm()) {
    os << "(FBMM ";
  } else {
    os << "(Unknown ";
  }
  if (s.fbfft()) {
    os << "FBFFT";
  } else if (s.cufft()) {
    os << "cuFFT";
  } else {
    os << "Unknown";
  }
  os << ") " << s.sizes;
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
  THCState* state,
  THCudaTensor* inputTH,
  THCudaTensor* inputComplexTH,
  THCudaTensor* inputComplexTrTH,
  THCudaTensor* inputComplexBufferTH,
  cufftHandle* plan,
  cufftHandle* planInverse) {
  if (convPass_.pass == ConvolutionPass::kUpdateOutput) {
    B_ = torchToDeviceTensor<float, 4>(state, inputTH);
    BComplex_ = torchToDeviceTensor<float, 5>(state, inputComplexTH);
    BComplexT_ = torchToDeviceTensor<float, 5>(state, inputComplexTrTH);
    if (inputComplexBufferTH) {
      BComplexBuffer_ =
        torchToDeviceTensor<float, 5>(state, inputComplexBufferTH);
    }
    fftPlanB_ = plan;
    fftPlanInverseB_ = planInverse;
  } else if (convPass_.pass == ConvolutionPass::kAccGradParameters) {
    A_ = torchToDeviceTensor<float, 4>(state, inputTH);
    AComplex_ = torchToDeviceTensor<float, 5>(state, inputComplexTH);
    AComplexT_ = torchToDeviceTensor<float, 5>(state, inputComplexTrTH);
    if (inputComplexBufferTH) {
      AComplexBuffer_ =
        torchToDeviceTensor<float, 5>(state, inputComplexBufferTH);
    }
    fftPlanA_ = plan;
    fftPlanInverseA_ = planInverse;
  } else {
    DCHECK_EQ(ConvolutionPass::kUpdateGradInput, convPass_.pass);
    C_ = torchToDeviceTensor<float, 4>(state, inputTH);
    CComplex_ = torchToDeviceTensor<float, 5>(state, inputComplexTH);
    CComplexT_ = torchToDeviceTensor<float, 5>(state, inputComplexTrTH);
    if (inputComplexBufferTH) {
      CComplexBuffer_ =
        torchToDeviceTensor<float, 5>(state, inputComplexBufferTH);
    }
    fftPlanC_ = plan;
  }
  return *this;
}

CuFFTConvolution&
CuFFTConvolution::withOutputAndBuffers(
  THCState* state,
  THCudaTensor* outputTH,
  THCudaTensor* outputComplexTH,
  THCudaTensor* outputComplexTrTH,
  THCudaTensor* outputComplexBufferTH,
  cufftHandle* plan,
  cufftHandle* planInverse) {
  if (convPass_.pass == ConvolutionPass::kUpdateOutput) {
    C_ = torchToDeviceTensor<float, 4>(state, outputTH);
    CComplex_ = torchToDeviceTensor<float, 5>(state, outputComplexTH);
    CComplexT_ = torchToDeviceTensor<float, 5>(state, outputComplexTrTH);
    if (outputComplexBufferTH) {
      CComplexBuffer_ =
        torchToDeviceTensor<float, 5>(state, outputComplexBufferTH);
    }
    fftPlanC_ = plan;
  } else if (convPass_.pass == ConvolutionPass::kAccGradParameters) {
    B_ = torchToDeviceTensor<float, 4>(state, outputTH);
    BComplex_ = torchToDeviceTensor<float, 5>(state, outputComplexTH);
    BComplexT_ = torchToDeviceTensor<float, 5>(state, outputComplexTrTH);
    if (outputComplexBufferTH) {
      BComplexBuffer_ =
        torchToDeviceTensor<float, 5>(state, outputComplexBufferTH);
    }
    fftPlanB_ = plan;
    fftPlanInverseB_ = planInverse;
  } else {
    DCHECK_EQ(ConvolutionPass::kUpdateGradInput, convPass_.pass);
    B_ = torchToDeviceTensor<float, 4>(state, outputTH);
    BComplex_ = torchToDeviceTensor<float, 5>(state, outputComplexTH);
    BComplexT_ = torchToDeviceTensor<float, 5>(state, outputComplexTrTH);
    if (outputComplexBufferTH) {
      BComplexBuffer_ =
        torchToDeviceTensor<float, 5>(state, outputComplexBufferTH);
    }
    fftPlanB_ = plan;
    fftPlanInverseB_ = planInverse;
  }
  return *this;
}

CuFFTConvolution&
CuFFTConvolution::withFiltersAndBuffers(
  THCState* state,
  THCudaTensor* filtersTH,
  THCudaTensor* filtersComplexTH,
  THCudaTensor* filtersComplexTrTH,
  THCudaTensor* filtersComplexBufferTH,
  cufftHandle* plan,
  cufftHandle* planInverse) {
  if (convPass_.pass == ConvolutionPass::kUpdateOutput) {
    A_ = torchToDeviceTensor<float, 4>(state, filtersTH);
    AComplex_ = torchToDeviceTensor<float, 5>(state, filtersComplexTH);
    AComplexT_ = torchToDeviceTensor<float, 5>(state, filtersComplexTrTH);
    if (filtersComplexBufferTH) {
      AComplexBuffer_ =
        torchToDeviceTensor<float, 5>(state, filtersComplexBufferTH);
    }
    fftPlanA_ = plan;
    fftPlanInverseA_ = planInverse;
  } else if (convPass_.pass == ConvolutionPass::kAccGradParameters) {
    C_ = torchToDeviceTensor<float, 4>(state, filtersTH);
    CComplex_ = torchToDeviceTensor<float, 5>(state, filtersComplexTH);
    CComplexT_ = torchToDeviceTensor<float, 5>(state, filtersComplexTrTH);
    if (filtersComplexBufferTH) {
      CComplexBuffer_ =
        torchToDeviceTensor<float, 5>(state, filtersComplexBufferTH);
    }
    fftPlanC_ = plan;
  } else {
    DCHECK_EQ(ConvolutionPass::kUpdateGradInput, convPass_.pass);
    A_ = torchToDeviceTensor<float, 4>(state, filtersTH);
    AComplex_ = torchToDeviceTensor<float, 5>(state, filtersComplexTH);
    AComplexT_ = torchToDeviceTensor<float, 5>(state, filtersComplexTrTH);
    if (filtersComplexBufferTH) {
      AComplexBuffer_ =
        torchToDeviceTensor<float, 5>(state, filtersComplexBufferTH);
    }
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
  if (strategy_->batchmm()) {
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
  } else if (strategy_->manymm()) {
    // If you want this version you must set up streams properly
    CuFFTConvolutionMxMMany();
  } else if (strategy_->fbmm()) {
    // A, B and C nomenclature is relative to cuBLAS' column-major.
    // In row-major fbmm, this is reversed.
    if (convPass_.pass == ConvolutionPass::kUpdateOutput) {
      transposeMM<5, false, true>(
        BComplex_, AComplex_, CComplex_, norm_.x, getStream(0));
    } else if (convPass_.pass == ConvolutionPass::kUpdateGradInput) {
      transposeMM<5, false, false>(
        BComplex_, AComplex_, CComplex_, norm_.x, getStream(0));
    } else if (convPass_.pass == ConvolutionPass::kAccGradParameters) {
      transposeMM<5, true, false>(
        BComplex_, AComplex_, CComplex_, norm_.x, getStream(0));
    } else {
      throw std::runtime_error("Invalid pass for CuFFTConvolution");
    }
  } else {
    throw std::runtime_error("Invalid MM mode for CuFFTConvolution");
  }
}

void CuFFTConvolution::run() {
  if (!strategy_) {
    strategy_ = &CuFFTStrategy::defaultStrategy();
  }

  DCHECK(nullptr != A_.data());
  DCHECK(nullptr != B_.data());
  DCHECK(nullptr != C_.data());
  DCHECK(nullptr != AComplex_.data());
  DCHECK(nullptr != BComplex_.data());
  DCHECK(nullptr != CComplex_.data());
  DCHECK(strategy_->fbmm() || nullptr != AComplexT_.data());
  DCHECK(strategy_->fbmm() || nullptr != BComplexT_.data());
  DCHECK(strategy_->fbmm() || nullptr != CComplexT_.data());

  VLOG(2) << A_ << " @ " << A_.data()
          << AComplex_ << " @ " << AComplex_.data()
          << AComplexT_ << " @ " << AComplexT_.data();

  VLOG(2) << B_ << " @ " << B_.data()
          << BComplex_ << " @ " << BComplex_.data()
          << BComplexT_ << " @ " << BComplexT_.data();

  VLOG(2) << C_ << " @ " << C_.data()
          << CComplex_ << " @ " << CComplex_.data()
          << CComplexT_ << " @ " << CComplexT_.data();

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

  if (strategy_->cufft()) {
    DCHECK_EQ(A_.getSize(2), B_.getSize(2));
    DCHECK_EQ(A_.getSize(3), B_.getSize(3));
    DCHECK_EQ(C_.getSize(2), B_.getSize(2));
    DCHECK_EQ(C_.getSize(3), B_.getSize(3));
  }

  norm_ = (strategy_->cufft()) ?
    make_cuComplex(scale_ / (A_.getSize(2) * A_.getSize(3)), 0.0f) :
    make_cuComplex(scale_ / (AComplex_.getSize(3) * AComplex_.getSize(3)),
                   0.0f);

  cudaStream_t s0 = getStream(0);
  cudaStream_t s1 = getStream(1);
  auto handle0 = getCircular(cublasHandles_, 0);
  // FFT followed by transpose in same stream
  if (strategy_->cufft()) {
    fft2d<2>(A_, AComplex_, FFTParameters().forward(), fftPlanA_, s0);
  } else {
    auto dA = A_.template downcastOuter<3>();
    auto dAC = AComplex_.template downcastOuter<4>();
    facebook::cuda::DeviceTensor<float, 4> dACB;
    facebook::cuda::DeviceTensor<float, 4>* pdACB;
    if (AComplexBuffer_.data()) {
      dACB = AComplexBuffer_.template downcastOuter<4>();
      pdACB = &dACB;
    } else {
      pdACB = nullptr;
    }
    fbfft2dHost<1>(dA,
                   dAC,
                   pdACB,
                   FFTParameters().withFbfft().forward(),
                   s0);
  }

  if (!strategy_->fbmm()) {
    // Transpose A_ (? ? y x) -> (y x ? ?) (row-major formulation)
    transposeAsComplex(AComplex_, AComplexT_, 2, handle0, s0);
  }

  auto handle1 = getCircular(cublasHandles_, 1);
  // FFT followed by transpose in same stream
  if (strategy_->cufft()) {
    fft2d<2>(B_, BComplex_, FFTParameters().forward(), fftPlanB_, s1);
  } else {
    auto dB = B_.template downcastOuter<3>();
    auto dBC = BComplex_.template downcastOuter<4>();
    facebook::cuda::DeviceTensor<float, 4> dBCB;
    facebook::cuda::DeviceTensor<float, 4>* pdBCB;
    if (BComplexBuffer_.data()) {
      dBCB = BComplexBuffer_.template downcastOuter<4>();
      pdBCB = &dBCB;
    } else {
      pdBCB = nullptr;
    }
    fbfft2dHost<1>(dB,
                   dBC,
                   pdBCB,
                   FFTParameters().withFbfft().forward(),
                   s1);
  }

  if (!strategy_->fbmm()) {
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
  }

  if (strategy_->manymm()) {
    allWaitOnK(0);
    allWaitOnK(1);
  } else {
    kWaitsOnL(0, 1);
  }

  // Run MxM
  CuFFTConvolutionMxM();
  if (strategy_->manymm()) {
    kWaitsOnAll(0);
  }

  cudaStream_t s = getStream(0);
  if (!strategy_->fbmm()) {
    auto handle = getCircular(cublasHandles_, 0);
    // Transpose followed by IFFT in same stream s0 as the MxM
    // Transpose input (y x ? ?) -> (? ? y x) (row-major formulation)
    transposeAsComplex(CComplexT_, CComplex_, 2, handle, s);
  }
  if (strategy_->cufft()) {
    fft2d<2>(C_, CComplex_, FFTParameters().inverse().normalize(false),
           fftPlanC_, s);
  } else {
    auto dC = C_.template downcastOuter<3>();
    auto dCC = CComplex_.template downcastOuter<4>();
    facebook::cuda::DeviceTensor<float, 4> dCCB;
    facebook::cuda::DeviceTensor<float, 4>* pdCCB;
    if (CComplexBuffer_.data()) {
      dCCB = CComplexBuffer_.template downcastOuter<4>();
      pdCCB = &dCCB;
    } else {
      pdCCB = nullptr;
    }
    fbfft2dHost<1>(dC,
                   dCC,
                   pdCCB,
                   FFTParameters().withFbfft().inverse().normalize(false),
                   s);
  }
  // Everything is transitively synchronized on stream0 at this point
}

} } } // namespace

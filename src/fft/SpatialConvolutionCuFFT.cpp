// Copyright 2014 Facebook

#include "THCTensor.h"
#include "DeviceTensorUtils.h"
#include "CuFFTConvolution.cuh"
#include "CuFFTConvolution_UpdateOutput.cuh"
#include "CuFFTConvolution_AccGradParameters.cuh"
#include "CuFFTConvolution_UpdateGradInput.cuh"
#include "CuFFTStrategy.h"
#include "CuFFTWrapper.cuh"
#include "Utils.h"
#include "util/Misc.h"

#include <folly/Format.h>
#include <folly/ScopeGuard.h>
#include <unordered_map>
#include <memory>
#include <vector>

using namespace std;
using namespace facebook::CUDAUtil;
using namespace facebook::deeplearning::torch;

namespace facebook { namespace deeplearning { namespace torch {

namespace detail {
// unique_ptr destructor for device-malloc'd memory
typedef cufftHandle cufftPlan;
struct CuFFTPlanDeleter {
  void operator()(cufftHandle* p) {
    if (p && *p) {
      THCudaCheck(cudaDeviceSynchronize());
      if (CUFFT_SUCCESS != cufftDestroy(*p)) {
        throw runtime_error(folly::format(
          "cufftDestroy error when freeing {}", *p).str().c_str());
      }
      delete p;
    }
  }
};

enum struct BufferType : int {
  InputBuffer = 1,
    WeightBuffer,
    OutputBuffer,
    SentinelStartData,
    InputReal,
    InputReal2,
    WeightReal,
    WeightReal2,
    OutputReal,
    OutputReal2,
    SentinelReal,
    InputComplex,
    InputComplex2,
    WeightComplex,
    WeightComplex2,
    OutputComplex,
    OutputComplex2,
    SentinelComplex,
    InputComplexTr,
    InputComplexTr2,
    WeightComplexTr,
    WeightComplexTr2,
    OutputComplexTr,
    OutputComplexTr2,
    SentinelComplexTranspose,
    Sentinel
    };


// Type shortnames
auto inputBufferType = BufferType::InputBuffer;
auto weightBufferType = BufferType::WeightBuffer;
auto outputBufferType = BufferType::OutputBuffer;

auto inputCType = BufferType::InputComplex;
auto weightCType = BufferType::WeightComplex;
auto outputCType = BufferType::OutputComplex;
auto inputCTrType = BufferType::InputComplexTr;
auto weightCTrType = BufferType::WeightComplexTr;
auto outputCTrType = BufferType::OutputComplexTr;

auto inputCType2 = BufferType::InputComplex2;
auto weightCType2 = BufferType::WeightComplex2;
auto outputCType2 = BufferType::OutputComplex2;
auto inputCTrType2 = BufferType::InputComplexTr2;
auto weightCTrType2 = BufferType::WeightComplexTr2;
auto outputCTrType2 = BufferType::OutputComplexTr2;

// 2-D FFTs only atm
const int kFFTDims = 2;

template <int FFTDim> class CuFFTBuffers {
 public:
  int currentDevice() {
    int currentDevice;
    THCudaCheck(cudaGetDevice(&currentDevice));
    return currentDevice;
  }

  THCudaTensor* buffer(
    THCState* state,
    const vector<long>& tensorSizes,
      BufferType bt) {
    DCHECK_LE(1, (int)bt);
    DCHECK_GT((int)BufferType::SentinelComplexTranspose, (int)bt);
    // In the non-cufft case, never allocate real buffers, just complex buffers
    assert((int)BufferType::SentinelReal <= (int)bt ||
           (int)bt <= (int)BufferType::SentinelStartData);
    TupleFFTBufferType key((BufferIntType)bt, currentDevice());
    auto th = makeCuFFTTensorComplex<FFTDim>(
      state, tensorSizes, (bufferMap_[key]).get());

    if (bufferMap_.count(key) > 0) {
      // At this point storage is refcounted if reused, save to delete.
      bufferMap_.erase(key);
    }
    bufferMap_.emplace(key, std::move(th));
    return bufferMap_[key].get();
  }

  THCudaTensor* cufftBuffer(
    THCState* state,
    THCudaTensor* real,
    const vector<long>& maxSizes,
    BufferType bt) {
    DCHECK_LE(1, (int)bt);
    DCHECK_GT((int)BufferType::SentinelComplexTranspose, (int)bt);
    TupleFFTBufferType key((BufferIntType)bt, currentDevice());
    auto th = ((int)bt < (int)BufferType::SentinelReal) ?
      makeCuFFTTensorReal<FFTDim>(
        state, real, maxSizes, (bufferMap_[key]).get())
      :
      makeCuFFTTensorComplex<FFTDim>(
        state, real, maxSizes, (bufferMap_[key]).get());

    if (bufferMap_.count(key) > 0) {
      // At this point storage is refcounted if reused, save to delete.
      bufferMap_.erase(key);
    }
    bufferMap_.emplace(key, std::move(th));
    return bufferMap_[key].get();
  }

  // cudaDeviceProp.concurrentKernels always returns 1 for some reason but we
  // really want 16 streams
  static constexpr int kNumConcurrentKernels = 16;

  vector<cudaStream_t> streams() {
    CudaDeviceType key(currentDevice());
    if (streams_.count(key) > 0) {
      DCHECK_EQ(1, streams_.count(key));
      return streams_[key];
    }

    streams_.emplace(key, vector<cudaStream_t>());
    auto& res = streams_[key];

    const cudaDeviceProp& prop = getDeviceProperties(currentDevice());
    // Fun fact, prop.concurrentKernels always returns 1 and screws up
    // performance. The actual number we want is 16 for Kepler.
    constexpr int kNumStreams = kNumConcurrentKernels;
    for (auto i = 0; i < kNumStreams; ++i) {
      res.push_back(cudaStream_t());
      auto err = cudaStreamCreate(&(res.back()));
      DCHECK_EQ(CUBLAS_STATUS_SUCCESS, err);
    }
    DCHECK_EQ(kNumStreams, res.size());
    return res;
  }

  vector<cudaEvent_t> events() {
    CudaDeviceType key(currentDevice());
    if (events_.count(key) > 0) {
      DCHECK_EQ(1, events_.count(key));
      return events_[key];
    }

    events_.emplace(key, vector<cudaEvent_t>());
    auto& res = events_[key];

    const cudaDeviceProp& prop = getDeviceProperties(currentDevice());
    // Fun fact, prop.concurrentKernels always returns 1 and screws up
    // performance. The actual number we want is 16 for Kepler.
    constexpr int kNumEvents = kNumConcurrentKernels;
    for (auto i = 0; i < kNumEvents; ++i) {
      res.push_back(cudaEvent_t());
      auto err = cudaEventCreate(&(res.back()));
      DCHECK_EQ(CUBLAS_STATUS_SUCCESS, err);
    }
    DCHECK_EQ(kNumEvents, res.size());
    return res;
  }

  vector<cublasHandle_t> handles() {
    CudaDeviceType key(currentDevice());
    if (cublasHandles_.count(key) > 0) {
      DCHECK_EQ(1, cublasHandles_.count(key));
      return cublasHandles_[key];
    }

    cublasHandles_.emplace(key, vector<cublasHandle_t>());
    auto& res = cublasHandles_[key];

    // Fun fact, prop.concurrentKernels always returns 1 and screws up
    // performance. The actual number we want is 16 for Kepler.
    constexpr int kNumCublasHandles = kNumConcurrentKernels;
    for (auto i = 0; i < kNumCublasHandles; ++i) {
      res.push_back(cublasHandle_t());
      auto err = cublasCreate(&(res.back()));
      DCHECK_EQ(CUBLAS_STATUS_SUCCESS, err);
    }
    DCHECK_EQ(kNumCublasHandles, res.size());
    return res;
  }

  // Stream index is needed to disambiguate plans of FFTs that run in parallel
  // on different streams: they cannot reuse the same plane
  cufftPlan* plan(THCState* state,
                  THCudaTensor* realTH,
                  THCudaTensor* complexTH,
                  FFTParameters params,
                  uint32_t streamInd) {
    // Only 2 parallel FFTs at any point.
    DCHECK_GE(1, streamInd);
    TensorSizeType sizes(THCudaTensor_size(state, realTH, 0),
                         THCudaTensor_size(state, realTH, 1),
                         THCudaTensor_size(state, realTH, 2),
                         THCudaTensor_size(state, realTH, 3));
    TensorStrideType strides(THCudaTensor_stride(state, realTH, 0),
                             THCudaTensor_stride(state, realTH, 1),
                             THCudaTensor_stride(state, realTH, 2),
                             THCudaTensor_stride(state, realTH, 3));
    // Even if we don't use the streamInd to bind a plan to a stream here, we
    // still need to pass it to distringuish between concurrent FFTs.
    // This disambiguates the risk of running concurrent FFTs with the same
    // plan in case they have the same hash.
    TupleFFTType key(sizes, strides, params.forwardFFT(),
                     currentDevice(), streamInd);
    if (cufftPlanMap_.count(key) > 0) {
      DCHECK_EQ(1, cufftPlanMap_.count(key)); // collisions not allowed
      return cufftPlanMap_[key].get();
    }

    auto p = makeCuFFTPlan<2, 4>(
      torchToDeviceTensor<float, 4>(state, realTH),
      torchToDeviceTensor<float, 5>(state, complexTH),
      params);
    auto h = folly::make_unique<cufftPlan, CuFFTPlanDeleter>(p);
    cufftPlanMap_.emplace(key, std::move(h));
    return cufftPlanMap_[key].get();
  }

  THCudaTensor* bufferWriteOnly(THCState* state,
                                int s1, int s2, int s3, int s4) {
    TensorSizeType sizes(s1, s2, s3, s4);
    TupleWriteOnlyType key(sizes, currentDevice());
    if (writeOnlyTensorMap_.count(key) > 0) {
      DCHECK_EQ(1, writeOnlyTensorMap_.count(key)); // collisions not allowed
      return writeOnlyTensorMap_[key].get();
    }
    auto strideCols = s4;
    auto size = vector<long>({s1, s2, s3, s4});
    auto stride = vector<long>({size[1] * size[2] * strideCols,
          size[2] * strideCols,
          strideCols,
          1});
    auto h = makeTHCudaTensorFull(state, size, stride);
    // Always fill with zeros
    THCudaTensor_fill(state, h.get(), 0.0f);
    writeOnlyTensorMap_.emplace(key, std::move(h));
    return writeOnlyTensorMap_[key].get();
  }

  THCudaTensor* bufferWriteOnlyPadded(
    THCState* state,
    THCudaTensor* t, int s1, int s2, int s3, int s4) {
    TensorSizeType sizes(s1, s2, s3, s4);
    TensorStrideType strides(THCudaTensor_stride(state, t, 0),
                             THCudaTensor_stride(state, t, 1),
                             THCudaTensor_stride(state, t, 2),
                             THCudaTensor_stride(state, t, 3));
    TupleWriteOnlyPaddedType key(sizes, strides, currentDevice());
    if (writeOnlyPaddedTensorMap_.count(key) > 0) {
      DCHECK_EQ(1, writeOnlyPaddedTensorMap_.count(key)); // no collisions
      return writeOnlyPaddedTensorMap_[key].get();
    }
    vector<long> sz({s1, s2, s3, s4});
    vector<long> st({
        THCudaTensor_stride(state, t, 0),
          THCudaTensor_stride(state, t, 1),
          THCudaTensor_stride(state, t, 2),
          THCudaTensor_stride(state, t, 3)});
    auto h = makeAliasedTHCudaTensorFull(state, t, sz, st);
    writeOnlyPaddedTensorMap_.emplace(key, std::move(h));
    return writeOnlyPaddedTensorMap_[key].get();
  }

  static CuFFTBuffers& singleton() {
    static CuFFTBuffers bufs;
    return bufs;
  }

  void reset() {
    THCudaCheck(cudaDeviceSynchronize());

    bufferMap_.clear();

    destroyVectorMap(cublasHandles_, cublasDestroy, cudaSuccess);
    destroyVectorMap(events_, cudaEventDestroy, cudaSuccess);
    destroyVectorMap(streams_, cudaStreamDestroy, cudaSuccess);

    cufftPlanMap_.clear();
    writeOnlyTensorMap_.clear();
    writeOnlyPaddedTensorMap_.clear();
  }

 private:
  CuFFTBuffers(const CuFFTBuffers&) = delete;
  CuFFTBuffers& operator=(CuFFTBuffers&) = delete;

  template<class T, class U, class W>
  void destroyVectorMap(T& t, U u, W w) {
    for (auto& kvp : t) {
      for (auto& h : kvp.second) {
        DCHECK_EQ(w, u(h));
      }
      kvp.second.clear();
    }
    t.clear();
  }

  ~CuFFTBuffers() {
    reset();
  }

  CuFFTBuffers() { }

  typedef int CudaDeviceType;

  // Reusable streams for parallel execution, kNumConcurrentKernels per device
  unordered_map<CudaDeviceType, vector<cudaStream_t>> streams_;

  // Reusable events for parallel execution, kNumConcurrentKernels per device
  unordered_map<CudaDeviceType, vector<cudaEvent_t>> events_;

  // Reusable cublas contexts, one per device
  unordered_map<CudaDeviceType,
                     vector<cublasHandle_t>> cublasHandles_;

  typedef bool FFTForwardType;
  typedef int StreamIndType;
  typedef tuple<long, long, long, long> TensorSizeType;
  typedef tuple<long, long, long, long> TensorStrideType;
  typedef tuple<TensorSizeType,
                     TensorStrideType,
                     FFTForwardType,
                     CudaDeviceType,
                     StreamIndType> TupleFFTType;
  unordered_map<TupleFFTType, unique_ptr<
    cufftPlan, CuFFTPlanDeleter>> cufftPlanMap_;

  // Output buffers, contiguous to avoid synchronizations in subsequent passes
  typedef tuple<TensorSizeType, CudaDeviceType> TupleWriteOnlyType;
  unordered_map<TupleWriteOnlyType, unique_ptr<
    THCudaTensor, CudaTensorDeleter>> writeOnlyTensorMap_;

  // Output buffers, non-contiguous correspond to actual padded FFT sizes
  typedef tuple<TensorSizeType, TensorStrideType,
                     CudaDeviceType> TupleWriteOnlyPaddedType;
  unordered_map<TupleWriteOnlyPaddedType, unique_ptr<
    THCudaTensor, CudaTensorDeleter>> writeOnlyPaddedTensorMap_;

  // Reusable buffers for holding intermediate FFTs as well as intermediate
  // transpositions of FFTs
  typedef int BufferIntType;
  typedef tuple<BufferIntType, CudaDeviceType> TupleFFTBufferType;
  unordered_map<TupleFFTBufferType, unique_ptr<
    THCudaTensor, CudaTensorDeleter>> bufferMap_;
};

#define MAKE_INPUT_BUFFER(LUA_BUFFER)                 \
  makeCuFFTTensorComplex<kFFTDims>(state,             \
    fftp.makeComplexTensorSizes<true>(numBatches,     \
                                      numInputPlanes, \
                                      probSizes[0],   \
                                      probSizes[1]),  \
    LUA_BUFFER);

#define MAKE_OUTPUT_BUFFER(LUA_BUFFER)                 \
  makeCuFFTTensorComplex<kFFTDims>(state,              \
    fftp.makeComplexTensorSizes<true>(numBatches,      \
                                      numOutputPlanes, \
                                      probSizes[0],    \
                                      probSizes[1]),   \
    LUA_BUFFER);

#define MAKE_WEIGHT_BUFFER(LUA_BUFFER)                 \
  makeCuFFTTensorComplex<kFFTDims>(state,              \
    fftp.makeComplexTensorSizes<true>(numOutputPlanes, \
                                      numInputPlanes,  \
                                      probSizes[0],    \
                                      probSizes[1]),   \
    LUA_BUFFER);

void updateOutputTH(THCState* state,
                    const THParams& p,
                    const ProblemSizes& originalSizes,
                    const CuFFTStrategy& s) {
  THCudaTensor* input = p.input;
  THCudaTensor* weight = p.weight;
  THCudaTensor* output = p.output;
  THCudaTensor* bias = p.bias;

  THCudaTensor_resize4d(state,
                        output,
                        originalSizes.batchSize,
                        originalSizes.filterSize,
                        originalSizes.outputSizeRow,
                        originalSizes.outputSizeCol);

  // 2-D FFT atm
  CHECK_EQ(4, THCudaTensor_nDimension(state, input));
  auto& buffers = CuFFTBuffers<kFFTDims>::singleton();

  // Always 4-D when passed to CUDA
  DCHECK_GE(s.sizes.inputSizeRow, THCudaTensor_size(state, weight, 2));
  DCHECK_GE(s.sizes.inputSizeCol, THCudaTensor_size(state, weight, 3));

  auto numBatches = THCudaTensor_size(state, input, 0);
  auto numInputPlanes = THCudaTensor_size(state, input, 1);
  auto numOutputPlanes = THCudaTensor_size(state, output, 1);

  // Enlarged buffer for FFTs
  const auto& probSizes = s.sizes.sizes();

  auto oTmp = (s.fbfft()) ?
    output :
    buffers.bufferWriteOnly(state,
                            s.sizes.batchSize,
                            s.sizes.filterSize,
                            s.sizes.rows(),
                            s.sizes.cols());

  // The following has to happen if not properly padded from above ... :/
  // No synchronization since we reuse a buffer, however we still need to copy
  auto weightR = (s.fbfft()) ?
    weight :
    buffers.cufftBuffer(state, weight, probSizes, BufferType::WeightReal);
  // See D1581014, to avoid an extra copy, the capacity must have allocated
  // with makeTHCudaTensorFull when using cufft
  auto inputR = (s.fbfft()) ?
    input :
    buffers.cufftBuffer(state, input, probSizes, BufferType::InputReal);

  auto fftp = FFTParameters().forward();
  if (s.fbfft()) fftp = fftp.withFbfft();

  auto inputCPtr = MAKE_INPUT_BUFFER(p.buffers.input);
  auto inputC = inputCPtr.get();
  DCHECK_EQ(p.buffers.input->storage, inputC->storage);

  auto outputCPtr = MAKE_OUTPUT_BUFFER(p.buffers.output);
  auto outputC = outputCPtr.get();
  DCHECK_EQ(p.buffers.output->storage, outputC->storage);

  auto weightCPtr = MAKE_WEIGHT_BUFFER(p.buffers.weight);
  auto weightC = weightCPtr.get();
  DCHECK_EQ(p.buffers.weight->storage, weightC->storage);

  auto inputCTrPtr = MAKE_INPUT_BUFFER(p.buffers.inputTranspose);
  auto inputCTr = inputCTrPtr.get();
  DCHECK_EQ(p.buffers.inputTranspose->storage, inputCTr->storage);

  auto outputCTrPtr = MAKE_OUTPUT_BUFFER(p.buffers.outputTranspose);
  auto outputCTr = outputCTrPtr.get();
  DCHECK_EQ(p.buffers.outputTranspose->storage, outputCTr->storage);

  auto weightCTrPtr = MAKE_WEIGHT_BUFFER(p.buffers.weightTranspose);
  auto weightCTr = weightCTrPtr.get();
  DCHECK_EQ(p.buffers.weightTranspose->storage, weightCTr->storage);

  // Plans
  auto planInput = (s.fbfft()) ?
    nullptr :
    buffers.plan(state, inputR, inputC, FFTParameters().forward(), 1);
  auto planWeight = (s.fbfft()) ?
    nullptr :
    buffers.plan(state, weightR, weightC, FFTParameters().forward(), 0);
  auto planOutput = (s.fbfft()) ?
    nullptr :
    buffers.plan(state, oTmp, outputC,
                 FFTParameters().inverse().normalize(false), 0);

  auto inputCBuffer = (s.fbfft()) ?
    buffers.buffer(state,
                   fftp.makeComplexTensorSizes<true>(numBatches,
                                                     numInputPlanes,
                                                     probSizes[0],
                                                     probSizes[1]),
                   inputBufferType) :
    nullptr;
  auto weightCBuffer = (s.fbfft()) ?
    buffers.buffer(state,
                   fftp.makeComplexTensorSizes<true>(numOutputPlanes,
                                                     numInputPlanes,
                                                     probSizes[0],
                                                     probSizes[1]),
                   weightBufferType) :
    nullptr;
  auto outputCBuffer = (s.fbfft()) ?
    buffers.buffer(state,
                   fftp.makeComplexTensorSizes<false>(numBatches,
                                                      numOutputPlanes,
                                                      probSizes[0],
                                                      probSizes[1]),
                   outputBufferType) :
    nullptr;

  // Handles
  auto handles = buffers.handles();

  if (s.cufft()) {
    // Sanity checks
    DCHECK_EQ(THCudaTensor_size(state, inputC, 2), s.sizes.rows());
    DCHECK_EQ(THCudaTensor_size(state, inputC, 3),
              numHermitian(s.sizes.cols()));
    DCHECK_EQ(THCudaTensor_size(state, weightC, 2), s.sizes.rows());
    DCHECK_EQ(THCudaTensor_size(state, weightC, 3),
              numHermitian(s.sizes.cols()));
    DCHECK_EQ(THCudaTensor_size(state, outputC, 2), s.sizes.rows());
    DCHECK_EQ(THCudaTensor_size(state, outputC, 3),
              numHermitian(s.sizes.cols()));
  }

  if (s.fbfft()) {
    // Sanity checks
    DCHECK_EQ(THCudaTensor_size(state, inputC, 3), s.sizes.rows());
    DCHECK_EQ(THCudaTensor_size(state, inputC, 2),
              numHermitian(s.sizes.cols()));
    DCHECK_EQ(THCudaTensor_size(state, weightC, 3), s.sizes.rows());
    DCHECK_EQ(THCudaTensor_size(state, weightC, 2),
              numHermitian(s.sizes.cols()));
    DCHECK_EQ(THCudaTensor_size(state, outputC, 3), s.sizes.rows());
    DCHECK_EQ(THCudaTensor_size(state, outputC, 2),
              numHermitian(s.sizes.cols()));
  }

  // Actual run
  CuFFTConvolution conv(ConvolutionPass(ConvolutionPass::kUpdateOutput));
  conv.withInputAndBuffers(
    state, inputR, inputC, inputCTr, inputCBuffer, planInput)
    .withFiltersAndBuffers(
      state, weightR, weightC, weightCTr, weightCBuffer, planWeight)
    .withOutputAndBuffers(state,
                          oTmp, outputC, outputCTr, outputCBuffer, planOutput)
    .withStreams(buffers.streams())
    .withEvents(buffers.events())
    .withCuBLASHandles(handles)
    .withStrategy(&s);
  CuFFTConvolution_UpdateOutput(state, &conv, oTmp, bias);

  // Barrier, asynchronous from the host PoV
  cudaEvent_t e = conv.getEvent(0);
  if (e) {
    THCudaCheck(cudaEventRecord(e, conv.getStream(0)));
    THCudaCheck(cudaStreamWaitEvent(nullptr, e, 0));
  }

  // Only cufft forces us to create multiple tensors and pad
  if (s.cufft()) {
    // Padded buffer from FFT with enlarged strides but final sizes
    auto oTmp2 = buffers.bufferWriteOnlyPadded(state,
                                               oTmp,
                                               s.sizes.batchSize,
                                               s.sizes.filterSize,
                                               s.sizes.outputSizeRow,
                                               s.sizes.outputSizeCol);

    // See D1581014, storage capacity is larger, we can resize to remove padding
    // resize4d is ok performance-wise since lua reuses buffer too
    THCudaTensor_resize4d(state,
                          output,
                          originalSizes.batchSize,
                          originalSizes.filterSize,
                          originalSizes.outputSizeRow,
                          originalSizes.outputSizeCol);
    THCudaTensor_copy(state, output, oTmp2);
  }
}


void updateGradInputTH(THCState* state,
                       const THParams& p,
                       const ProblemSizes& originalSizes,
                       const CuFFTStrategy& s) {
  THCudaTensor* gradInput = p.input;
  THCudaTensor* weight = p.weight;
  THCudaTensor* gradOutput = p.output;

  THCudaTensor_resize4d(state,
                        gradInput,
                        originalSizes.batchSize,
                        originalSizes.planeSize,
                        originalSizes.inputSizeRow,
                        originalSizes.inputSizeCol);

  // 2-D FFT atm
  CHECK_EQ(4, THCudaTensor_nDimension(state, gradOutput));
  auto& buffers = CuFFTBuffers<kFFTDims>::singleton();

  // Always 4-D when passed to CUDA
  DCHECK_GE(s.sizes.inputSizeRow, THCudaTensor_size(state, weight, 2));
  DCHECK_GE(s.sizes.inputSizeCol, THCudaTensor_size(state, weight, 3));
  DCHECK_GE(s.sizes.inputSizeRow, THCudaTensor_size(state, gradOutput, 2));
  DCHECK_GE(s.sizes.inputSizeCol, THCudaTensor_size(state, gradOutput, 3));

  auto numBatches = THCudaTensor_size(state, gradInput, 0);
  auto numInputPlanes = THCudaTensor_size(state, gradInput, 1);
  auto numOutputPlanes = THCudaTensor_size(state, gradOutput, 1);

  // Enlarged buffer for FFTs
  const auto& probSizes = s.sizes.sizes();
  auto giTmp = (s.fbfft()) ?
    gradInput :
    buffers.bufferWriteOnly(state,
                            s.sizes.batchSize,
                            s.sizes.planeSize,
                            s.sizes.rows(),
                            s.sizes.cols());

  // The following has to happen if not properly padded from above ... :/
  auto weightR = (s.fbfft()) ?
    weight :
    buffers.cufftBuffer(state, weight, probSizes, BufferType::WeightReal);
  // See D1581014, to avoid an extra copy, the capacity must have allocated
  // with makeTHCudaTensorFull
  auto gradOutputR = (s.fbfft()) ?
    gradOutput :
    buffers.cufftBuffer(state, gradOutput, probSizes, BufferType::OutputReal);

  auto fftp = FFTParameters().forward();
  if (s.fbfft()) fftp = fftp.withFbfft();

  auto gradInputCPtr = MAKE_INPUT_BUFFER(p.buffers.input);
  auto gradInputC = gradInputCPtr.get();
  DCHECK_EQ(p.buffers.input->storage, gradInputC->storage);

  auto gradOutputCPtr = MAKE_OUTPUT_BUFFER(p.buffers.output);
  auto gradOutputC = gradOutputCPtr.get();
  DCHECK_EQ(p.buffers.output->storage, gradOutputC->storage);

  auto weightCPtr = MAKE_WEIGHT_BUFFER(p.buffers.weight);
  auto weightC = weightCPtr.get();
  DCHECK_EQ(p.buffers.weight->storage, weightC->storage);

  auto gradInputCTrPtr = MAKE_INPUT_BUFFER(p.buffers.inputTranspose);
  auto gradInputCTr = gradInputCTrPtr.get();
  DCHECK_EQ(p.buffers.inputTranspose->storage, gradInputCTr->storage);

  auto gradOutputCTrPtr = MAKE_OUTPUT_BUFFER(p.buffers.outputTranspose);
  auto gradOutputCTr = gradOutputCTrPtr.get();
  DCHECK_EQ(p.buffers.outputTranspose->storage, gradOutputCTr->storage);

  auto weightCTrPtr = MAKE_WEIGHT_BUFFER(p.buffers.weightTranspose);
  auto weightCTr = weightCTrPtr.get();
  DCHECK_EQ(p.buffers.weightTranspose->storage, weightCTr->storage);

  auto gradInputCBuffer = (s.fbfft()) ?
    buffers.buffer(state,
                   fftp.makeComplexTensorSizes<false>(numBatches,
                                                      numInputPlanes,
                                                      probSizes[0],
                                                      probSizes[1]),
                   inputBufferType) :
    nullptr;
  auto weightCBuffer = (s.fbfft()) ?
    buffers.buffer(state,
                   fftp.makeComplexTensorSizes<true>(numOutputPlanes,
                                                     numInputPlanes,
                                                     probSizes[0],
                                                     probSizes[1]),
                   weightBufferType) :
    nullptr;
  auto gradOutputCBuffer = (s.fbfft()) ?
    buffers.buffer(state,
                   fftp.makeComplexTensorSizes<true>(numBatches,
                                                     numOutputPlanes,
                                                     probSizes[0],
                                                     probSizes[1]),
                   outputBufferType) :
    nullptr;

  // Plans
  auto planInput = (s.fbfft()) ?
    nullptr :
    buffers.plan(
      state, giTmp, gradInputC,
      FFTParameters().inverse().normalize(false), 0);
  auto planWeight = (s.fbfft()) ?
    nullptr :
    buffers.plan(state, weightR, weightC,
                 FFTParameters().forward(), 1);
  auto planOutput = (s.fbfft()) ?
    nullptr :
    buffers.plan(state, gradOutputR, gradOutputC,
                 FFTParameters().forward(), 0);

  // Handles
  auto handles = buffers.handles();

 // Actual run
  CuFFTConvolution conv(ConvolutionPass(ConvolutionPass::kUpdateGradInput));
  conv.withInputAndBuffers(
    state,
    giTmp, gradInputC, gradInputCTr, gradInputCBuffer, planInput)
    .withFiltersAndBuffers(
      state,
      weightR, weightC, weightCTr, weightCBuffer, planWeight)
    .withOutputAndBuffers(
      state,
      gradOutputR, gradOutputC, gradOutputCTr, gradOutputCBuffer, planOutput)
    .withStreams(buffers.streams())
    .withEvents(buffers.events())
    .withCuBLASHandles(handles)
    .withStrategy(&s);
  CuFFTConvolution_UpdateGradInput(&conv);

  // Barrier, asynchronous from the host PoV
  cudaEvent_t e = conv.getEvent(0);
  if (e) {
    THCudaCheck(cudaEventRecord(e, conv.getStream(0)));
    THCudaCheck(cudaStreamWaitEvent(nullptr, e, 0));
  }

  if (s.cufft()) {
    // Padded buffer from FFT with enlarged strides but final sizes
    auto giTmp2 = buffers.bufferWriteOnlyPadded(state,
                                                giTmp,
                                                s.sizes.batchSize,
                                                s.sizes.planeSize,
                                                s.sizes.inputSizeRow,
                                                s.sizes.inputSizeCol);

    // See D1581014, storage capacity is larger, we can resize to remove padding
    // resize4d is ok performance-wise since lua reuses buffer too
    THCudaTensor_resize4d(state,
                          gradInput,
                          originalSizes.batchSize,
                          originalSizes.planeSize,
                          originalSizes.inputSizeRow,
                          originalSizes.inputSizeCol);
    THCudaTensor_copy(state, gradInput, giTmp2);
  }
}

void accGradParametersTH(THCState* state,
                         const THParams& p,
                         const ProblemSizes& originalSizes,
                         const CuFFTStrategy& s) {
  THCudaTensor* input = p.input;
  THCudaTensor* gradWeight = p.weight;
  THCudaTensor* gradOutput = p.output;
  THCudaTensor* gradBias = p.bias;
  float scale = p.scale;

  THCudaTensor_resize4d(state,
                        gradWeight,
                        originalSizes.filterSize,
                        originalSizes.planeSize,
                        originalSizes.weightSizeRow,
                        originalSizes.weightSizeCol);

  // 2-D FFT atm
  CHECK_EQ(4, THCudaTensor_nDimension(state, input));
  auto& buffers = CuFFTBuffers<kFFTDims>::singleton();

  auto numBatches = THCudaTensor_size(state, input, 0);
  auto numInputPlanes = THCudaTensor_size(state, input, 1);
  auto numOutputPlanes = THCudaTensor_size(state, gradOutput, 1);

  // Enlarged buffer for FFTs
  const auto& probSizes = s.sizes.sizes();
  auto gwTmp = (s.fbfft()) ?
    gradWeight :
    buffers.bufferWriteOnly(state,
                            s.sizes.filterSize,
                            s.sizes.planeSize,
                            s.sizes.rows(),
                            s.sizes.cols());

  // See D1581014, to avoid an extra copy, the capacity must have allocated
  // with makeTHCudaTensorFull
  auto inputR = (s.fbfft()) ?
    input :
    buffers.cufftBuffer(state, input, probSizes, BufferType::InputReal);
  // See D1581014, to avoid an extra copy, the capacity must have allocated
  // with makeTHCudaTensorFull
  auto gradOutputR = (s.fbfft()) ?
    gradOutput :
    buffers.cufftBuffer(state, gradOutput, probSizes, BufferType::OutputReal);

  // Use properly padded real arrays as model for complex
  auto fftp = FFTParameters().forward();
  if (s.fbfft()) fftp = fftp.withFbfft();

  auto inputCPtr = MAKE_INPUT_BUFFER(p.buffers.input);
  auto inputC = inputCPtr.get();
  DCHECK_EQ(p.buffers.input->storage, inputC->storage);

  auto gradOutputCPtr = MAKE_OUTPUT_BUFFER(p.buffers.output);
  auto gradOutputC = gradOutputCPtr.get();
  DCHECK_EQ(p.buffers.output->storage, gradOutputC->storage);

  auto gradWeightCPtr = MAKE_WEIGHT_BUFFER(p.buffers.weight);
  auto gradWeightC = gradWeightCPtr.get();
  DCHECK_EQ(p.buffers.weight->storage, gradWeightC->storage);

  auto inputCTrPtr = MAKE_INPUT_BUFFER(p.buffers.inputTranspose);
  auto inputCTr = inputCTrPtr.get();
  DCHECK_EQ(p.buffers.inputTranspose->storage, inputCTr->storage);

  auto gradOutputCTrPtr = MAKE_OUTPUT_BUFFER(p.buffers.outputTranspose);
  auto gradOutputCTr = gradOutputCTrPtr.get();
  DCHECK_EQ(p.buffers.outputTranspose->storage, gradOutputCTr->storage);

  auto gradWeightCTrPtr = MAKE_WEIGHT_BUFFER(p.buffers.weightTranspose);
  auto gradWeightCTr = gradWeightCTrPtr.get();
  DCHECK_EQ(p.buffers.weightTranspose->storage, gradWeightCTr->storage);

  auto inputCBuffer = (s.fbfft()) ?
    buffers.buffer(state,
                   fftp.makeComplexTensorSizes<true>(numBatches,
                                                     numInputPlanes,
                                                     probSizes[0],
                                                     probSizes[1]),
                   inputBufferType) :
    nullptr;
  auto gradWeightCBuffer = (s.fbfft()) ?
    buffers.buffer(state,
                   fftp.makeComplexTensorSizes<false>(numOutputPlanes,
                                                      numInputPlanes,
                                                      probSizes[0],
                                                      probSizes[1]),
                   weightBufferType) :
    nullptr;
  auto gradOutputCBuffer = (s.fbfft()) ?
    buffers.buffer(state,
                   fftp.makeComplexTensorSizes<true>(numBatches,
                                                     numOutputPlanes,
                                                     probSizes[0],
                                                     probSizes[1]),
                   outputBufferType) :
    nullptr;

  DCHECK(THCudaTensor_isContiguous(state, inputC) > 0);
  DCHECK(THCudaTensor_isContiguous(state, gradOutputC) > 0);
  DCHECK(THCudaTensor_isContiguous(state, gradWeightC) > 0);
  DCHECK(THCudaTensor_isContiguous(state, inputCTr) > 0);
  DCHECK(THCudaTensor_isContiguous(state, gradOutputCTr) > 0);
  DCHECK(THCudaTensor_isContiguous(state, gradWeightCTr) > 0);

  auto planInput = (s.fbfft()) ?
    nullptr :
    buffers.plan(state, inputR, inputC, FFTParameters().forward(), 0);
  auto planWeight = (s.fbfft()) ?
    nullptr :
    buffers.plan(
      state, gwTmp, gradWeightC, FFTParameters().inverse().normalize(false), 0);
  auto planOutput = (s.fbfft()) ?
    nullptr :
    buffers.plan(state, gradOutputR, gradOutputC, FFTParameters().forward(), 1);

  auto handles = buffers.handles();
  CuFFTConvolution conv(ConvolutionPass(ConvolutionPass::kAccGradParameters));
  conv.withInputAndBuffers(
    state, inputR, inputC, inputCTr, inputCBuffer, planInput)
    .withFiltersAndBuffers(
      state, gwTmp, gradWeightC, gradWeightCTr, gradWeightCBuffer, planWeight)
    .withOutputAndBuffers(
      state,
      gradOutputR, gradOutputC, gradOutputCTr, gradOutputCBuffer, planOutput)
    .withStreams(buffers.streams())
    .withEvents(buffers.events())
    .withCuBLASHandles(handles)
    .withScale(scale)
    .withStrategy(&s);
  CuFFTConvolution_AccGradParameters(
    state, &conv, gradOutputR, gradBias, scale);

  // Barrier, asynchronous from the host PoV
  cudaEvent_t e = conv.getEvent(0);
  if (e) {
    THCudaCheck(cudaEventRecord(e, conv.getStream(0)));
    THCudaCheck(cudaStreamWaitEvent(nullptr, e, 0));
  }

  if (s.cufft()) {
    // Padded buffer from FFT with enlarged strides but final sizes
    auto gwTmp2 = buffers.bufferWriteOnlyPadded(state,
                                                gwTmp,
                                                s.sizes.filterSize,
                                                s.sizes.planeSize,
                                                s.sizes.weightSizeRow,
                                                s.sizes.weightSizeCol);

    // See D1581014, storage capacity is larger, we can resize to remove padding
    // resize4d is ok performance-wise since lua reuses buffer too
    THCudaTensor_resize4d(state,
                          gradWeight,
                          originalSizes.filterSize,
                          originalSizes.planeSize,
                          originalSizes.weightSizeRow,
                          originalSizes.weightSizeCol);
    THCudaTensor_copy(state, gradWeight, gwTmp2);
  }
}

void cleanupBuffers() {
  cudaDeviceSynchronize();
  auto& buffers = CuFFTBuffers<kFFTDims>::singleton();
  buffers.reset();
  ConvolutionPass dummy(ConvolutionPass::kUpdateOutput);
  CuFFTConvolution conv(dummy);
  conv.reset();
  cudaDeviceSynchronize();
}
}

}}}  // namespaces

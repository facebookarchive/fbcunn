// Copyright 2014 Facebook

#include "DeviceTensorUtils.h"
#include "THCTensor.h"
#include "fft/CuFFTConvolution_UpdateOutput.cuh"
#include "fft/CuFFTConvolution_AccGradParameters.cuh"
#include "fft/CuFFTConvolution_UpdateGradInput.cuh"
#include "fft/CuFFTWrapper.cuh"
#include "fft/CuFFTConvolution.h"
#include "fft/CuFFTConvolution.cuh"
#include "fft/Utils.h"
#include "util/Misc.h"

#include <folly/Format.h>
#include <folly/ScopeGuard.h>
#include <unordered_map>
#include <memory>
#include <vector>

using namespace std;
using namespace facebook::CUDAUtil;

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
  InputReal = 1,
    WeightReal,
    OutputReal,
    SentinelReal,
    InputComplex,
    WeightComplex,
    OutputComplex,
    SentinelComplex,
    InputComplexTr,
    WeightComplexTr,
    OutputComplexTr,
    SentinelComplexTranspose,
    Sentinel
    };

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
      THCudaTensor* real,
      const vector<long>& maxSizes,
      BufferType bt) {
    DCHECK_LE(1, (int)bt);
    DCHECK_GT((int)BufferType::SentinelComplexTranspose, (int)bt);
    TupleFFTBufferType key((BufferIntType)bt, currentDevice());
    auto th = ((int)bt < (int)BufferType::SentinelReal) ?
      makeCuFFTTensorReal<FFTDim>(
        real, maxSizes, (bufferMap_[key]).get())
      :
      makeCuFFTTensorComplex<FFTDim>(
        real, maxSizes, (bufferMap_[key]).get());

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
  cufftPlan* plan(THCudaTensor* realTH,
                  THCudaTensor* complexTH,
                  FFTParameters params,
                  uint32_t streamInd) {
    // Only 2 parallel FFTs at any point.
    DCHECK_GE(1, streamInd);
    TensorSizeType sizes(THCudaTensor_size(NULL, realTH, 0),
                         THCudaTensor_size(NULL, realTH, 1),
                         THCudaTensor_size(NULL, realTH, 2),
                         THCudaTensor_size(NULL, realTH, 3));
    TensorStrideType strides(THCudaTensor_stride(NULL, realTH, 0),
                             THCudaTensor_stride(NULL, realTH, 1),
                             THCudaTensor_stride(NULL, realTH, 2),
                             THCudaTensor_stride(NULL, realTH, 3));
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

    auto p = makeCuFFTPlan<2, 4>(torchToDeviceTensor<float, 4>(realTH),
                                 torchToDeviceTensor<float, 5>(complexTH),
                                 params);
    auto h = folly::make_unique<cufftPlan, CuFFTPlanDeleter>(p);
    cufftPlanMap_.emplace(key, std::move(h));
    return cufftPlanMap_[key].get();
  }

  THCudaTensor* bufferWriteOnly(int s1, int s2, int s3, int s4) {
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
    auto h = makeTHCudaTensorFull(size, stride);
    // Always fill with zeros
    THCudaTensor_fill(NULL, h.get(), 0.0f);
    writeOnlyTensorMap_.emplace(key, std::move(h));
    return writeOnlyTensorMap_[key].get();
  }

  THCudaTensor* bufferWriteOnlyPadded(
    THCudaTensor* t, int s1, int s2, int s3, int s4) {
    TensorSizeType sizes(s1, s2, s3, s4);
    TensorStrideType strides(THCudaTensor_stride(NULL, t, 0),
                             THCudaTensor_stride(NULL, t, 1),
                             THCudaTensor_stride(NULL, t, 2),
                             THCudaTensor_stride(NULL, t, 3));
    TupleWriteOnlyPaddedType key(sizes, strides, currentDevice());
    if (writeOnlyPaddedTensorMap_.count(key) > 0) {
      DCHECK_EQ(1, writeOnlyPaddedTensorMap_.count(key)); // no collisions
      return writeOnlyPaddedTensorMap_[key].get();
    }
    vector<long> sz({s1, s2, s3, s4});
    vector<long> st({
        THCudaTensor_stride(NULL, t, 0),
          THCudaTensor_stride(NULL, t, 1),
          THCudaTensor_stride(NULL, t, 2),
          THCudaTensor_stride(NULL, t, 3)});
    auto h = makeAliasedTHCudaTensorFull(t, sz, st);
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

void updateOutputTH(const THParams& p,
                    const ProblemSizes& originalSizes,
                    const CuFFTStrategy& s) {
  THCudaTensor* input = p.input;
  THCudaTensor* weight = p.weight;
  THCudaTensor* output = p.output;
  THCudaTensor* bias = p.bias;

  // 2-D FFT atm
  CHECK_EQ(4, THCudaTensor_nDimension(NULL, input));
  auto& buffers = CuFFTBuffers<kFFTDims>::singleton();

  // Always 4-D when passed to CUDA
  DCHECK_GE(s.sizes.inputSizeRow, THCudaTensor_size(NULL, weight, 2));
  DCHECK_GE(s.sizes.inputSizeCol, THCudaTensor_size(NULL, weight, 3));

  // Setup types based on strategy, determines which buffers get reused
  auto inputCType = BufferType::InputComplex;
  auto weightCType = BufferType::WeightComplex;
  auto outputCType = BufferType::OutputComplex;
  auto inputCTrType = BufferType::InputComplexTr;
  auto weightCTrType = BufferType::WeightComplexTr;
  auto outputCTrType = BufferType::OutputComplexTr;

  // Enlarged buffer for FFTs
  const auto& probSizes = s.sizes.sizes();
  auto oTmp = buffers.bufferWriteOnly(s.sizes.batchSize,
                                      s.sizes.filterSize,
                                      s.sizes.rows(),
                                      s.sizes.cols());

  // The following has to happen if not properly padded from above ... :/
  // No synchronization since we reuse a buffer, however we still need to copy
  auto weightR =
    buffers.buffer(weight, probSizes, BufferType::WeightReal);
  // See D1581014, to avoid an extra copy, the capacity must have allocated
  // with makeTHCudaTensorFull
  auto inputR =
    buffers.buffer(input, probSizes, BufferType::InputReal);

  // Use properly padded real arrays as model for complex
  auto inputC =
    buffers.buffer(inputR, probSizes, inputCType);
  auto weightC =
    buffers.buffer(weightR, probSizes, weightCType);
  auto outputC = buffers.buffer(oTmp, probSizes, outputCType);
  auto inputCTr = buffers.buffer(inputR, probSizes, inputCTrType);
  auto weightCTr = buffers.buffer(weightR, probSizes, weightCTrType);
  auto outputCTr = buffers.buffer(oTmp, probSizes, outputCTrType);

  // Plans
  auto planInput = buffers.plan(inputR, inputC, FFTParameters().forward(), 1);
  auto planWeight = buffers.plan(
    weightR, weightC, FFTParameters().forward(), 0);
  auto planOutput = buffers.plan(
    oTmp, outputC, FFTParameters().inverse().normalize(false), 0);

  // Handles
  auto handles = buffers.handles();

  // Sanity checks
  DCHECK_EQ(THCudaTensor_size(NULL, inputC, 2), s.sizes.rows());
  DCHECK_EQ(THCudaTensor_size(NULL, inputC, 3),
            numHermitianSymmetryCols(s.sizes.cols()));
  DCHECK_EQ(THCudaTensor_size(NULL, weightC, 2), s.sizes.rows());
  DCHECK_EQ(THCudaTensor_size(NULL, weightC, 3),
            numHermitianSymmetryCols(s.sizes.cols()));
  DCHECK_EQ(THCudaTensor_size(NULL, outputC, 2), s.sizes.rows());
  DCHECK_EQ(THCudaTensor_size(NULL, outputC, 3),
            numHermitianSymmetryCols(s.sizes.cols()));

  // Actual run
  CuFFTConvolution conv(ConvolutionPass(ConvolutionPass::kUpdateOutput));
  conv.withInputAndBuffers(inputR, inputC, inputCTr, planInput)
    .withFiltersAndBuffers(weightR, weightC, weightCTr, planWeight)
    .withOutputAndBuffers(oTmp, outputC, outputCTr, planOutput)
    .withStreams(buffers.streams())
    .withEvents(buffers.events())
    .withCuBLASHandles(handles)
    .withStrategy(&s);
  CuFFTConvolution_UpdateOutput(&conv, oTmp, bias);

  // Barrier, asynchronous from the host PoV
  cudaEvent_t e = conv.getEvent(0);
  if (e) {
    THCudaCheck(cudaEventRecord(e, conv.getStream(0)));
    THCudaCheck(cudaStreamWaitEvent(nullptr, e, 0));
  }

  // Padded buffer from FFT with enlarged strides but final sizes
  auto oTmp2 = buffers.bufferWriteOnlyPadded(oTmp,
                                             s.sizes.batchSize,
                                             s.sizes.filterSize,
                                             s.sizes.outputSizeRow,
                                             s.sizes.outputSizeCol);

  // See D1581014, storage capacity is larger, we can resize to remove padding
  // resize4d is ok performance-wise since lua reuses buffer too
  THCudaTensor_resize4d(NULL, output,
                        originalSizes.batchSize,
                        originalSizes.filterSize,
                        originalSizes.outputSizeRow,
                        originalSizes.outputSizeCol);
  THCudaTensor_copy(NULL, output, oTmp2);
}


void updateGradInputTH(const THParams& p,
                       const ProblemSizes& originalSizes,
                       const CuFFTStrategy& s) {
  THCudaTensor* gradInput = p.input;
  THCudaTensor* weight = p.weight;
  THCudaTensor* gradOutput = p.output;

  // 2-D FFT atm
  CHECK_EQ(4, THCudaTensor_nDimension(NULL, gradOutput));
  auto& buffers = CuFFTBuffers<kFFTDims>::singleton();

  // Always 4-D when passed to CUDA
  DCHECK_GE(s.sizes.inputSizeRow, THCudaTensor_size(NULL, weight, 2));
  DCHECK_GE(s.sizes.inputSizeCol, THCudaTensor_size(NULL, weight, 3));
  DCHECK_GE(s.sizes.inputSizeRow, THCudaTensor_size(NULL, gradOutput, 2));
  DCHECK_GE(s.sizes.inputSizeCol, THCudaTensor_size(NULL, gradOutput, 3));

  // Setup types based on strategy, determines which buffers get reused
  auto inputCType = BufferType::InputComplex;
  auto weightCType = BufferType::WeightComplex;
  auto outputCType = BufferType::OutputComplex;
  auto inputCTrType = BufferType::InputComplexTr;
  auto weightCTrType = BufferType::WeightComplexTr;
  auto outputCTrType = BufferType::OutputComplexTr;

  // Enlarged buffer for FFTs
  const auto& probSizes = s.sizes.sizes();
  auto giTmp = buffers.bufferWriteOnly(s.sizes.batchSize,
                                       s.sizes.planeSize,
                                       s.sizes.rows(),
                                       s.sizes.cols());

  // The following has to happen if not properly padded from above ... :/
  auto weightR = buffers.buffer(
    weight, probSizes, BufferType::WeightReal);
  // See D1581014, to avoid an extra copy, the capacity must have allocated
  // with makeTHCudaTensorFull
  auto gradOutputR = buffers.buffer(
    gradOutput, probSizes, BufferType::OutputReal);

  // Use properly padded real arrays as model for complex
  auto gradInputC =
    buffers.buffer(giTmp, probSizes, inputCType);
  auto weightC =
    buffers.buffer(weightR, probSizes, weightCType);
  auto gradOutputC =
    buffers.buffer(gradOutputR, probSizes, outputCType);
  auto gradInputCTr = buffers.buffer(giTmp, probSizes, inputCTrType);
  auto weightCTr = buffers.buffer(weightR, probSizes, weightCTrType);
  auto gradOutputCTr = buffers.buffer(gradOutputR, probSizes, outputCTrType);

  // Plans
  auto planInput = buffers.plan(
    giTmp, gradInputC, FFTParameters().inverse().normalize(false), 0);
  auto planWeight = buffers.plan(
    weightR, weightC, FFTParameters().forward(), 1);
  auto planOutput = buffers.plan(
    gradOutputR, gradOutputC, FFTParameters().forward(), 0);

  // Handles
  auto handles = buffers.handles();

 // Actual run
  CuFFTConvolution conv(ConvolutionPass(ConvolutionPass::kUpdateGradInput));
  conv.withInputAndBuffers(giTmp, gradInputC, gradInputCTr, planInput)
    .withFiltersAndBuffers(weightR, weightC, weightCTr, planWeight)
    .withOutputAndBuffers(gradOutputR, gradOutputC, gradOutputCTr, planOutput)
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

  // Padded buffer from FFT with enlarged strides but final sizes
  auto giTmp2 = buffers.bufferWriteOnlyPadded(giTmp,
                                              s.sizes.batchSize,
                                              s.sizes.planeSize,
                                              s.sizes.inputSizeRow,
                                              s.sizes.inputSizeCol);

  // See D1581014, storage capacity is larger, we can resize to remove padding
  // resize4d is ok performance-wise since lua reuses buffer too
  THCudaTensor_resize4d(NULL, gradInput,
                        originalSizes.batchSize,
                        originalSizes.planeSize,
                        originalSizes.inputSizeRow,
                        originalSizes.inputSizeCol);
  THCudaTensor_copy(NULL, gradInput, giTmp2);
}

void accGradParametersTH(const THParams& p,
                         const ProblemSizes& originalSizes,
                         const CuFFTStrategy& s) {
  THCudaTensor* input = p.input;
  THCudaTensor* gradWeight = p.weight;
  THCudaTensor* gradOutput = p.output;
  THCudaTensor* gradBias = p.bias;
  float scale = p.scale;

  // 2-D FFT atm
  CHECK_EQ(4, THCudaTensor_nDimension(NULL, input));
  auto& buffers = CuFFTBuffers<kFFTDims>::singleton();

  auto inputCType = BufferType::InputComplex;
  auto weightCType = BufferType::WeightComplex;
  auto outputCType = BufferType::OutputComplex;
  auto inputCTrType = BufferType::InputComplexTr;
  auto weightCTrType = BufferType::WeightComplexTr;
  auto outputCTrType = BufferType::OutputComplexTr;

  // Enlarged buffer for FFTs
  const auto& probSizes = s.sizes.sizes();
  auto gwTmp = buffers.bufferWriteOnly(s.sizes.filterSize,
                                       s.sizes.planeSize,
                                       s.sizes.rows(),
                                       s.sizes.cols());

  // See D1581014, to avoid an extra copy, the capacity must have allocated
  // with makeTHCudaTensorFull
  auto inputR = buffers.buffer(input, probSizes, BufferType::InputReal);
  // See D1581014, to avoid an extra copy, the capacity must have allocated
  // with makeTHCudaTensorFull
  auto gradOutputR = buffers.buffer(
    gradOutput, probSizes, BufferType::OutputReal);

  // Use properly padded real arrays as model for complex
  auto inputC =
    buffers.buffer(inputR, probSizes, inputCType);
  auto gradWeightC =
    buffers.buffer(gwTmp, probSizes, weightCType);
  auto gradOutputC =
    buffers.buffer(gradOutputR, probSizes, outputCType);
  auto inputCTr = buffers.buffer(inputR, probSizes, inputCTrType);
  auto gradWeightCTr = buffers.buffer(gwTmp, probSizes, weightCTrType);
  auto gradOutputCTr = buffers.buffer(gradOutputR, probSizes, outputCTrType);

  DCHECK(THCudaTensor_isContiguous(NULL, inputC) > 0);
  DCHECK(THCudaTensor_isContiguous(NULL, gradOutputC) > 0);
  DCHECK(THCudaTensor_isContiguous(NULL, gradWeightC) > 0);
  DCHECK(THCudaTensor_isContiguous(NULL, inputCTr) > 0);
  DCHECK(THCudaTensor_isContiguous(NULL, gradOutputCTr) > 0);
  DCHECK(THCudaTensor_isContiguous(NULL, gradWeightCTr) > 0);

  auto planInput = buffers.plan(inputR, inputC, FFTParameters().forward(), 0);
  auto planWeight = buffers.plan(
    gwTmp, gradWeightC, FFTParameters().inverse().normalize(false), 0);
  auto planOutput = buffers.plan(
    gradOutputR, gradOutputC, FFTParameters().forward(), 1);

  auto handles = buffers.handles();
  CuFFTConvolution conv(ConvolutionPass(ConvolutionPass::kAccGradParameters));
  conv.withInputAndBuffers(inputR, inputC, inputCTr, planInput)
    .withFiltersAndBuffers(gwTmp, gradWeightC, gradWeightCTr, planWeight)
    .withOutputAndBuffers(gradOutputR, gradOutputC, gradOutputCTr, planOutput)
    .withStreams(buffers.streams())
    .withEvents(buffers.events())
    .withCuBLASHandles(handles)
    .withScale(scale)
    .withStrategy(&s);
  CuFFTConvolution_AccGradParameters(&conv, gradOutputR, gradBias, scale);

  // Barrier, asynchronous from the host PoV
  cudaEvent_t e = conv.getEvent(0);
  if (e) {
    THCudaCheck(cudaEventRecord(e, conv.getStream(0)));
    THCudaCheck(cudaStreamWaitEvent(nullptr, e, 0));
  }

  // Padded buffer from FFT with enlarged strides but final sizes
  auto gwTmp2 = buffers.bufferWriteOnlyPadded(gwTmp,
                                              s.sizes.filterSize,
                                              s.sizes.planeSize,
                                              s.sizes.weightSizeRow,
                                              s.sizes.weightSizeCol);

  // See D1581014, storage capacity is larger, we can resize to remove padding
  // resize4d is ok performance-wise since lua reuses buffer too
  THCudaTensor_resize4d(NULL, gradWeight,
                        originalSizes.filterSize,
                        originalSizes.planeSize,
                        originalSizes.weightSizeRow,
                        originalSizes.weightSizeCol);
  THCudaTensor_copy(NULL, gradWeight, gwTmp2);
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

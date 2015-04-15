/**
 * Copyright 2014 Facebook
 * @author Michael Mathieu (myrhev@fb.com)
 */

#include <cstdio>
#include <float.h>

namespace facebook { namespace deeplearning { namespace torch {
namespace detail {

namespace {

#define DIVUP(a, b) (((a) + (b) - 1) / (b))
const int SHARED_MEM_MAX_SIZE = 49152;

const int MV_N_REDUCE = 8;
const int MV_BUFFER_SIZE = 128;
// assume : bias is contiguous
//          score last dim is contiguous
//          score has enough elements
//          score is initially filled with zeros
//          shared memory buffer size == input_size * blockDim.y
//          weight last dim is contiguous
//          input last dim is contiguous
//          mapping is contiguous
// blockIdx.x  : MV columns
// blockIdx.y  : minibatch
// threadIdx.x : MV line
__global__ void
updateOutputWithTargetMV(const float* input,
                         const float* weight,
                         const float* bias,
                         const float* mapping,
                         const float* n_class_in_cluster,
                         const float* class_start_indices,
                         const float* target,
                         const long input_stride0,
                         const long weight_stride0,
                         const long score_stride0,
                         long input_size,
                         float* score) {
  __shared__ float buffer[MV_BUFFER_SIZE];
  // align input and score to current sample in minibatch
  input += input_stride0 * blockIdx.y;
  score += score_stride0 * blockIdx.y;

  // get the indices corresponding the the target
  const int itarget = (int)(target[blockIdx.y] - 0.5f); // - 0.5 : 1based->0
  const int cluster_target = (int)(mapping[2*itarget] - 0.5f);
  const int iclass_start = (int)(class_start_indices[cluster_target] + 0.5f);
  const int cluster_size = (int)(n_class_in_cluster[cluster_target] + 0.5f);

  // get the bias and weight of the target cluster + correct line
  const int lineIdx = blockIdx.x;
  const int nLinesParallel = gridDim.x;

  // do matrix vector multiply :
  const int tidxx = threadIdx.x;
  //   loop over lines
  for (int iline = lineIdx; iline < cluster_size; iline += nLinesParallel) {
    const float* weight0 = weight + weight_stride0 * (iclass_start + iline);
    //   map
    __syncthreads();
    register float tmp = 0.f;
    for (int i = tidxx; i < input_size; i += MV_BUFFER_SIZE)
      tmp += input[i] * weight0[i];
    buffer[tidxx] = tmp;
    //   reduce
/*
    for (unsigned int stride = MV_BUFFER_SIZE >> 1; stride > 0; stride >>= 1) {
      __syncthreads();
      if (tidxx < stride)
        buffer[tidxx] += buffer[tidxx+stride];
    }
    if (tidxx == 0)
      score[iline] = buffer[0] + bias[iclass_start + iline];
*/
    tmp = 0.f;
    __syncthreads();
    if (tidxx < MV_BUFFER_SIZE / MV_N_REDUCE) {
      for (int i = tidxx * MV_N_REDUCE; i < (tidxx + 1) * MV_N_REDUCE; ++i)
        tmp += buffer[i];
      buffer[tidxx] = tmp;
    }
    __syncthreads();
    // store result
    if (tidxx == 0) {
      tmp = buffer[0];
#pragma unroll
      for (int i = 1; i < MV_BUFFER_SIZE / MV_N_REDUCE; ++i)
        tmp += buffer[i];
      score[iline] = tmp + bias[iclass_start + iline];
    }
  }
}

const int LSM_BUFFER_SIZE = 128;
// assume :
//   mapping is contiguous
//   logsum is contiguous
//   score is contiguous in the last dim
// blockIdx.x : minibatch
// blockIdx.y : cluster/class
// threadIdx.x : worker
__global__ void
updateOutputWithTargetLSM(const float* target,
                          const float* mapping,
                          const float* n_class_in_cluster,
                          const float* class_score,
                          float* class_logsum,
                          float* cluster_score,
                          float* cluster_logsum,
                          const long class_score_stride0,
                          const long cluster_score_stride0,
                          int n_clusters,
                          float* loss) {
  __shared__ float buffer[LSM_BUFFER_SIZE + 1];
  const int tidx = threadIdx.x;
  const int nthreads = blockDim.x;

  const int itarget = (int)(target[blockIdx.x] - 0.5f);
  const int cluster_target = (int)(mapping[2*itarget] - 0.5f);
  const int idx_in_cluster_target = (int)(mapping[2*itarget+1] - 0.5f);
  const int cluster_size = (int)(n_class_in_cluster[cluster_target] + 0.5f);

  const float *score;
  float *logsum, target_score;
  int N;
  if (blockIdx.y == 0) {
    score = cluster_score + blockIdx.x * cluster_score_stride0;
    logsum = cluster_logsum + blockIdx.x;
    N = n_clusters;
    target_score = score[cluster_target];
  } else {
    score = class_score + blockIdx.x * class_score_stride0;
    logsum = class_logsum + blockIdx.x;
    N = cluster_size;
    target_score = score[idx_in_cluster_target];
  }

  // get max (from nn.LogSoftMax code)
  //   map
  float vmax = -FLT_MAX;
  for (int i = tidx; i < N; i += nthreads) {
    float z = score[i];
    if (vmax < z)
      vmax = z;
  }
  buffer[tidx] = vmax;
  //   reduce
  for (unsigned int stride = nthreads >> 1; stride > 0; stride >>= 1) {
    __syncthreads();
    if ((tidx < stride) && (buffer[tidx] < buffer[tidx+stride]))
      buffer[tidx] = buffer[tidx+stride];
  }
  //   store it at last position in buffer
  if (tidx == 0) {
    float max_k = -FLT_MAX;
    if (max_k < buffer[0])
      max_k = buffer[0];
    buffer[LSM_BUFFER_SIZE] = max_k;
  }
  __syncthreads();

  // logadd
  //   map
  float max_k = buffer[LSM_BUFFER_SIZE];
  buffer[tidx] = 0;
  for (int i = tidx; i < N; i += nthreads)
    buffer[tidx] += expf(score[i] - max_k);
  //   reduce
  for (unsigned int stride = nthreads >> 1; stride > 0; stride >>= 1) {
    __syncthreads();
    if (tidx < stride)
      buffer[tidx] += buffer[tidx+stride];
  }
  //   write result
  if (tidx == 0) {
    float logsum_k = max_k + logf(buffer[0]);
    *logsum = logsum_k;
    atomicAdd(loss, logsum_k - target_score);
  }
}

// assume :
//   mapping is contiguous
//   logsum is contiguous
//   score is contiguous in the last dim
// blockIdx.x : minibatch
// blockIdx.y : cluster/class
// threadIdx.x : worker
__global__ void
updateGradInputLSM(const float* target,
                   const float* mapping,
                   const float* n_class_in_cluster,
                   float* class_score,
                   float* class_logsum,
                   float* cluster_score,
                   float* cluster_logsum,
                   const long class_score_stride0,
                   const long cluster_score_stride0,
                   int n_clusters) {
  const int tidx = threadIdx.x;
  const int nthreads = blockDim.x;

  const int itarget = (int)(target[blockIdx.x] - 0.5f);
  const int cluster_target = (int)(mapping[2*itarget] - 0.5f);
  const int idx_in_cluster_target = (int)(mapping[2*itarget+1] - 0.5f);
  const int cluster_size = (int)(n_class_in_cluster[cluster_target] + 0.5f);

  float *score, logsum_k, *target_score;
  int N;
  if (blockIdx.y == 0) {
    score = cluster_score + blockIdx.x * cluster_score_stride0;
    logsum_k = cluster_logsum[blockIdx.x];
    N = n_clusters;
    target_score = score + cluster_target;
  } else {
    score = class_score + blockIdx.x * class_score_stride0;
    logsum_k = class_logsum[blockIdx.x];
    N = cluster_size;
    target_score = score + idx_in_cluster_target;
  }

  for (int i = tidx; i < N; i += nthreads)
    score[i] = expf(score[i] - logsum_k);
  __syncthreads(); //TODO : not exactly needed
  if (tidx == 0)
    *target_score -= 1.f;
}

const int MV2_NLINES = 128;
__global__ void
updateGradInputMV(const float* score,
                  const float* weight,
                  const float* mapping,
                  const float* n_class_in_cluster,
                  const float* class_start_indices,
                  const float* target,
                  const long gradInput_stride0,
                  const long weight_stride0,
                  const long score_stride0,
                  int input_size,
                  float* gradInput) {
  // align input and score to current sample in minibatch
  gradInput += gradInput_stride0 * blockIdx.y;
  score += score_stride0 * blockIdx.y;

  // get the indices corresponding the the target
  const int itarget = (int)(target[blockIdx.y] - 0.5f); // - 0.5 : 1based->0
  const int cluster_target = (int)(mapping[2*itarget] - 0.5f);
  const int iclass_start = (int)(class_start_indices[cluster_target] + 0.5f);
  const int cluster_size = (int)(n_class_in_cluster[cluster_target] + 0.5f);

  // get the bias and weight of the target cluster + correct line
  const int colIdx = blockIdx.x * MV2_NLINES + threadIdx.x;
  const int nColParallel = gridDim.x * MV2_NLINES;

  //   loop over lines
  weight += weight_stride0 * iclass_start;
  for (int icol = colIdx; icol < input_size; icol += nColParallel) {
    const float* weight0 = weight + icol;
    //   map
    register float tmp = 0.f;
    for (int i = 0; i < cluster_size; ++i)
      tmp += score[i] * weight0[weight_stride0 * i];
    gradInput[icol] = tmp;
  }
}


__global__ void
accGradParameters(const float* mapping,
                  const float* n_class_in_cluster,
                  const float* class_start_indices,
                  const float* target,
                  const float* input,
                  const float* score,
                  const int input_size,
                  const long input_stride0,
                  const long score_stride0,
                  const long gradWeight_stride0,
                  const float scale,
                  float* gradWeight,
                  float* gradBias) {
  // select minibatch
  input += blockIdx.x * input_stride0;
  score += blockIdx.x * score_stride0;
  const int itarget = (int)(target[blockIdx.x] - 0.5f); // - 0.5 : 1based->0
  const int cluster_target = (int)(mapping[2*itarget] - 0.5f);
  const int iclass_start = (int)(class_start_indices[cluster_target] + 0.5f);
  const int cluster_size = (int)(n_class_in_cluster[cluster_target] + 0.5f);
  gradWeight += iclass_start * gradWeight_stride0;
  gradBias += iclass_start;

  // fill shared memory
  const int iline_stride = DIVUP(cluster_size, gridDim.y);
  const int iline_start = blockIdx.y * iline_stride;
  const int iline_end = min(iline_start + iline_stride, cluster_size);
  const int iline_n = iline_end - iline_start;
  const int tidx = threadIdx.y * blockDim.x + threadIdx.x;
  const int nthreads = blockDim.x * blockDim.y;
  extern __shared__ float shared_input[];
  float* shared_score = shared_input + input_size;
  for (int i = tidx; i < input_size; i += nthreads)
    shared_input[i] = input[i];
  for (int i = tidx; i < iline_n; i += nthreads)
    shared_score[i] = score[iline_start + i];
  gradBias += iline_start;
  gradWeight += iline_start * gradWeight_stride0;

  __syncthreads();
  // outer product
  for (int iline = threadIdx.y; iline < iline_n; iline += blockDim.y) {
    float* gradWeight0 = gradWeight + iline * gradWeight_stride0;
    register const float score_cur = scale * shared_score[iline];
    for (int icol = threadIdx.x; icol < input_size; icol += blockDim.x)
      atomicAdd(gradWeight0 + icol, score_cur * shared_input[icol]);
    if (threadIdx.x == 0)
      atomicAdd(gradBias + iline, score_cur);
  }
}

} // namespace

void launchUpdateOutputWithTargetKernel(
  cudaStream_t stream,
  const float* input,
  const float* class_weight,
  const float* class_bias,
  const float* mapping,
  const float* n_class_in_cluster,
  const float* class_start_indices,
  const float* target,
  const long* input_strides,
  const long* class_weight_strides,
  const long* class_score_strides,
  const long* cluster_score_strides,
  const long input_size,
  const long minibatch_size,
  const long n_max_class_per_cluster,
  const long n_clusters,
  float* class_score,
  float* class_logsum,
  float* cluster_score,
  float* cluster_logsum,
  float* output) {
  { // run MV
    const long n_lines_on_grid = 64; //TODO: tune
    dim3 threads(MV_BUFFER_SIZE);
    dim3 blocks(min((int)n_lines_on_grid, (int)n_max_class_per_cluster),
                minibatch_size);
  updateOutputWithTargetMV<<<blocks, threads, 0, stream>>>(
    input, class_weight, class_bias, mapping, n_class_in_cluster,
    class_start_indices, target, input_strides[0], class_weight_strides[0],
    class_score_strides[0], input_size, class_score);
  }
  { // run logsoftmax
    dim3 blocks(minibatch_size, 2);
    dim3 threads(LSM_BUFFER_SIZE);
    updateOutputWithTargetLSM<<<blocks, threads, 0, stream>>>(
      target, mapping, n_class_in_cluster, class_score, class_logsum,
      cluster_score, cluster_logsum, class_score_strides[0],
      cluster_score_strides[0], n_clusters, output);
  }
}

void launchUpdateGradInput(
  cudaStream_t stream,
  const float* class_weight,
  const float* mapping,
  const float* n_class_in_cluster,
  const float* class_start_indices,
  const float* target,
  const long* gradInput_strides,
  const long* class_weight_strides,
  const long* class_score_strides,
  const long* cluster_score_strides,
  const long input_size,
  const long minibatch_size,
  const long n_max_class_per_cluster,
  const long n_clusters,
  float* class_score,
  float* class_logsum,
  float* cluster_score,
  float* cluster_logsum,
  float* gradInput) {
  cudaMemsetAsync(gradInput, 0, input_size * minibatch_size * sizeof(float), stream);
  { // bprop in logsoftmax
    dim3 blocks(minibatch_size, 2);
    dim3 threads(128); //TODO: tune
    updateGradInputLSM<<<blocks, threads, 0, stream>>>(
      target, mapping, n_class_in_cluster, class_score, class_logsum,
      cluster_score, cluster_logsum, class_score_strides[0],
      cluster_score_strides[0], n_clusters);
  }
  {
    dim3 threads(MV2_NLINES);
    dim3 blocks(32, minibatch_size); //TODO: tune
    updateGradInputMV<<<blocks, threads, 0, stream>>>(
      class_score, class_weight, mapping, n_class_in_cluster,
      class_start_indices, target, gradInput_strides[0],
      class_weight_strides[0], class_score_strides[0],
      input_size, gradInput);
  }
}

void launchAccGradParameters(
  cudaStream_t stream,
  const float* class_score,
  const float* mapping,
  const float* n_class_in_cluster,
  const float* class_start_indices,
  const float* target,
  const float* input,
  const long* input_strides,
  const long* class_score_strides,
  const long* class_gradWeight_strides,
  const long input_size,
  const long minibatch_size,
  const long n_max_class_per_cluster,
  const float scale,
  float* class_gradWeight,
  float* class_gradBias) {
  dim3 blocks(minibatch_size, 4); //TODO: tune
  dim3 threads(32, 4);
  const size_t shared_mem_size =
    (input_size + DIVUP(n_max_class_per_cluster, blocks.y)) * sizeof(float);
  if (shared_mem_size > SHARED_MEM_MAX_SIZE) {
    printf("HSM: not enough shared memory. Reduce input size and/or number \
of class in the largest cluster\n");
    exit(0);
  }
  accGradParameters<<<blocks, threads, shared_mem_size, stream>>>(
    mapping, n_class_in_cluster, class_start_indices, target, input,
    class_score, input_size, input_strides[0], class_score_strides[0],
    class_gradWeight_strides[0], scale, class_gradWeight,
    class_gradBias);
}

}}}}

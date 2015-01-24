/**
 * Copyright 2014 Facebook
 * @author Michael Mathieu (myrhev@fb.com)
 */

#include <lua.hpp>
#include <TH.h>
#include "THC.h"
#include <luaT.h>

namespace facebook { namespace deeplearning { namespace torch {

namespace detail {

void launchUpdateOutputWithTargetKernel(
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
  long input_size,
  long minibatch_size,
  long n_max_class_per_cluster,
  long n_clusters,
  float* class_score,
  float* class_logsum,
  float* cluster_score,
  float* cluster_logsum,
  float* output);

void launchUpdateGradInput(
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
  float* gradInput);

void launchAccGradParameters(
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
  float* class_gradBias);

} // namespace detail

namespace {

inline THCudaTensor* getFieldCudaTensor(lua_State* L, int arg,
                                        const char* name) {
  return static_cast<THCudaTensor*>(luaT_getfieldcheckudata(
                                      L, arg, name, "torch.CudaTensor"));
}
inline THCudaTensor* getCudaTensor(lua_State* L, int arg) {
  return static_cast<THCudaTensor*>(luaT_checkudata(L, arg,
                                                    "torch.CudaTensor"));
}

int updateOutputWithTarget(lua_State* L) {
  auto class_weight        = getFieldCudaTensor(L, 1, "class_weight");
  auto class_bias          = getFieldCudaTensor(L, 1, "class_bias");
  auto class_score         = getFieldCudaTensor(L, 1, "class_score");
  auto cluster_score       = getFieldCudaTensor(L, 1, "cluster_score");
  auto class_logsum        = getFieldCudaTensor(L, 1, "class_logsum");
  auto cluster_logsum      = getFieldCudaTensor(L, 1, "cluster_logsum");
  auto mapping             = getFieldCudaTensor(L, 1, "mapping");
  auto n_class_in_cluster  = getFieldCudaTensor(L, 1, "n_class_in_cluster");
  auto class_start_indices = getFieldCudaTensor(L, 1, "class_start_indices");
  auto output              = getFieldCudaTensor(L, 1, "output");
  //auto unk_index  = luaGetFieldIfNumberChecked<long>(L, 1, "unk_index") - 1;
  auto input      = getCudaTensor(L, 2);
  auto target     = getCudaTensor(L, 3);

  auto n_class    = class_bias->size[0];
  auto batch_size = input->size[0];
  auto input_size = input->size[1];
  auto class_weight_strides = class_weight->stride;
  auto input_strides = input->stride;
  auto class_score_strides = class_score->stride;
  auto cluster_score_strides = cluster_score->stride;
  auto n_max_class_per_cluster = class_score->size[1];
  auto n_clusters = n_class_in_cluster->size[0];
  auto input_data = THCudaTensor_data(NULL, input);
  auto class_weight_data = THCudaTensor_data(NULL, class_weight);
  auto class_bias_data = THCudaTensor_data(NULL, class_bias);
  auto mapping_data = THCudaTensor_data(NULL, mapping);
  auto n_class_in_cluster_data = THCudaTensor_data(NULL, n_class_in_cluster);
  auto class_start_indices_data = THCudaTensor_data(NULL, class_start_indices);
  auto target_data = THCudaTensor_data(NULL, target);
  auto class_score_data = THCudaTensor_data(NULL, class_score);
  auto class_logsum_data = THCudaTensor_data(NULL, class_logsum);
  auto cluster_score_data = THCudaTensor_data(NULL, cluster_score);
  auto cluster_logsum_data = THCudaTensor_data(NULL, cluster_logsum);
  auto output_data = THCudaTensor_data(NULL, output);

  detail::launchUpdateOutputWithTargetKernel(
    input_data,
    class_weight_data,
    class_bias_data,
    mapping_data,
    n_class_in_cluster_data,
    class_start_indices_data,
    target_data,
    input_strides,
    class_weight_strides,
    class_score_strides,
    cluster_score_strides,
    input_size,
    batch_size,
    n_max_class_per_cluster,
    n_clusters,
    class_score_data,
    class_logsum_data,
    cluster_score_data,
    cluster_logsum_data,
    output_data);

  return 0;
}

int updateGradInput(lua_State* L) {
  auto class_weight        = getFieldCudaTensor(L, 1, "class_weight");
  auto class_score         = getFieldCudaTensor(L, 1, "class_score");
  auto cluster_score       = getFieldCudaTensor(L, 1, "cluster_score");
  auto class_logsum        = getFieldCudaTensor(L, 1, "class_logsum");
  auto cluster_logsum      = getFieldCudaTensor(L, 1, "cluster_logsum");
  auto mapping             = getFieldCudaTensor(L, 1, "mapping");
  auto n_class_in_cluster  = getFieldCudaTensor(L, 1, "n_class_in_cluster");
  auto class_start_indices = getFieldCudaTensor(L, 1, "class_start_indices");
  auto gradInput           = getFieldCudaTensor(L, 1, "gradInput");
  auto target     = getCudaTensor(L, 2);

  auto batch_size = gradInput->size[0];
  auto input_size = gradInput->size[1];
  auto class_weight_strides = class_weight->stride;
  auto class_score_strides = class_score->stride;
  auto cluster_score_strides = cluster_score->stride;
  auto gradInput_strides = gradInput->stride;
  auto n_max_class_per_cluster = class_score->size[1];
  auto n_clusters = n_class_in_cluster->size[0];
  auto class_weight_data = THCudaTensor_data(NULL, class_weight);
  auto mapping_data = THCudaTensor_data(NULL, mapping);
  auto n_class_in_cluster_data = THCudaTensor_data(NULL, n_class_in_cluster);
  auto class_start_indices_data = THCudaTensor_data(NULL, class_start_indices);
  auto target_data = THCudaTensor_data(NULL, target);
  auto class_score_data = THCudaTensor_data(NULL, class_score);
  auto class_logsum_data = THCudaTensor_data(NULL, class_logsum);
  auto cluster_score_data = THCudaTensor_data(NULL, cluster_score);
  auto cluster_logsum_data = THCudaTensor_data(NULL, cluster_logsum);
  auto gradInput_data = THCudaTensor_data(NULL, gradInput);

  detail::launchUpdateGradInput(
    class_weight_data,
    mapping_data,
    n_class_in_cluster_data,
    class_start_indices_data,
    target_data,
    gradInput_strides,
    class_weight_strides,
    class_score_strides,
    cluster_score_strides,
    input_size,
    batch_size,
    n_max_class_per_cluster,
    n_clusters,
    class_score_data,
    class_logsum_data,
    cluster_score_data,
    cluster_logsum_data,
    gradInput_data);

  return 0;
}

int accGradParameters(lua_State* L) {
  auto class_score         = getFieldCudaTensor(L, 1, "class_score");
  auto mapping             = getFieldCudaTensor(L, 1, "mapping");
  auto n_class_in_cluster  = getFieldCudaTensor(L, 1, "n_class_in_cluster");
  auto class_start_indices = getFieldCudaTensor(L, 1, "class_start_indices");
  auto class_gradWeight    = getFieldCudaTensor(L, 1, "class_grad_weight");
  auto class_gradBias      = getFieldCudaTensor(L, 1, "class_grad_bias");
  auto input  = getCudaTensor(L, 2);
  auto target = getCudaTensor(L, 3);
  auto scale  = lua_tonumber(L, 4);

  auto class_score_data = THCudaTensor_data(NULL, class_score);
  auto mapping_data = THCudaTensor_data(NULL, mapping);
  auto n_class_in_cluster_data = THCudaTensor_data(NULL, n_class_in_cluster);
  auto class_start_indices_data = THCudaTensor_data(NULL, class_start_indices);
  auto target_data = THCudaTensor_data(NULL, target);
  auto input_data = THCudaTensor_data(NULL, input);
  auto input_strides = input->stride;
  auto class_score_strides = class_score->stride;
  auto class_gradWeight_strides = class_gradWeight->stride;
  auto n_max_class_per_cluster = class_score->size[1];
  auto input_size = input->size[1];
  auto batch_size = input->size[0];
  auto class_gradWeight_data = THCudaTensor_data(NULL, class_gradWeight);
  auto class_gradBias_data = THCudaTensor_data(NULL, class_gradBias);

  detail::launchAccGradParameters(
    class_score_data,
    mapping_data,
    n_class_in_cluster_data,
    class_start_indices_data,
    target_data,
    input_data,
    input_strides,
    class_score_strides,
    class_gradWeight_strides,
    input_size,
    batch_size,
    n_max_class_per_cluster,
    scale,
    class_gradWeight_data,
    class_gradBias_data);

  return 0;
}

int accGradParameters_directUpdate(lua_State* L) {
  auto class_score  = getFieldCudaTensor(L, 1, "class_score");
  auto mapping      = getFieldCudaTensor(L, 1, "mapping");
  auto class_weight = getFieldCudaTensor(L, 1, "class_weight");
  auto class_bias   = getFieldCudaTensor(L, 1, "class_bias");
  auto n_class_in_cluster  = getFieldCudaTensor(L, 1, "n_class_in_cluster");
  auto class_start_indices = getFieldCudaTensor(L, 1, "class_start_indices");
  auto input  = getCudaTensor(L, 2);
  auto target = getCudaTensor(L, 3);
  auto scale  = lua_tonumber(L, 4);

  auto class_score_data = THCudaTensor_data(NULL, class_score);
  auto mapping_data = THCudaTensor_data(NULL, mapping);
  auto n_class_in_cluster_data = THCudaTensor_data(NULL, n_class_in_cluster);
  auto class_start_indices_data = THCudaTensor_data(NULL, class_start_indices);
  auto target_data = THCudaTensor_data(NULL, target);
  auto input_data = THCudaTensor_data(NULL, input);
  auto input_strides = input->stride;
  auto class_score_strides = class_score->stride;
  auto class_weight_strides = class_weight->stride;
  auto n_max_class_per_cluster = class_score->size[1];
  auto input_size = input->size[1];
  auto batch_size = input->size[0];
  auto class_weight_data = THCudaTensor_data(NULL, class_weight);
  auto class_bias_data = THCudaTensor_data(NULL, class_bias);

  detail::launchAccGradParameters(
    class_score_data,
    mapping_data,
    n_class_in_cluster_data,
    class_start_indices_data,
    target_data,
    input_data,
    input_strides,
    class_score_strides,
    class_weight_strides,
    input_size,
    batch_size,
    n_max_class_per_cluster,
    scale,
    class_weight_data,
    class_bias_data);

  return 0;
}


const luaL_Reg functions[] = {
  {"HSM_updateOutputWithTarget", updateOutputWithTarget},
  {"HSM_updateGradInput", updateGradInput},
  {"HSM_accGradParameters", accGradParameters},
  {"HSM_accGradParameters_directUpdate", accGradParameters_directUpdate},
  {nullptr, nullptr},
};

} // namespace

void initHSMCuda(lua_State* L) {
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, functions, "nn");
  lua_pop(L, 1);
}


}}}

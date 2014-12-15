// Copyright 2004-present Facebook. All Rights Reserved.
// Author: Michael Mathieu <myrhev@fb.com>

#include <algorithm>
#include <cstdio>
#include <memory>
#include <limits>

#include <lua.hpp>
#include <mkl.h>
#include <luaT.h>
#include <omp.h>

#include "Blas.h"
#include "torch/fb/fbcunn/layers/LuaUtils.h"
#include "torch/fb/fbcunn/layers/Tensor.h"
#include "Vml.h"
#include <folly/Format.h>

namespace facebook {
namespace deeplearning {
namespace torch {

namespace {

template <class T> using thOps = thpp::detail::TensorOps<T>;

template <class T>
int updateOutputWithTarget(lua_State* L) {
  auto class_weight   = luaGetFieldIfTensorChecked<T>(L, 1, "class_weight");
  auto class_bias     = luaGetFieldIfTensorChecked<T>(L, 1, "class_bias");
  auto cluster_bias   = luaGetFieldIfTensorChecked<T>(L, 1, "cluster_bias");
  auto class_score    = luaGetFieldIfTensorChecked<T>(L, 1, "class_score");
  auto class_logsum   = luaGetFieldIfTensorChecked<T>(L, 1, "class_logsum");
  auto mapping        = luaGetFieldIfTensorChecked<long>(L, 1, "mapping");
  auto n_class_in_cluster =
    luaGetFieldIfTensorChecked<long>(L, 1, "n_class_in_cluster");
  auto class_start_indices =
    luaGetFieldIfTensorChecked<long>(L, 1, "class_start_indices");
  auto input      = luaGetTensorChecked<T>(L, 2);
  auto target     = luaGetTensorChecked<long>(L, 3);
  auto n_clusters = cluster_bias.size(0);
  auto n_class    = class_bias.size(0);
  auto batch_size = input.size(0);
  if (input.ndims() == 1)
    batch_size = 1;

  T output = 0.;
  long n_valid = 0;
  T loss;
  for (int i_batch = 0; i_batch < batch_size; ++i_batch) {
    long itarget = target.at({i_batch}) - 1; // 1based->0based
    long cluster_target = mapping.at({itarget, 0}) - 1; // 1based->0based
    long idx_in_cluster_target = mapping.at({itarget, 1}) - 1; // 1based->0based
    long cluster_size = n_class_in_cluster.at({cluster_target});
    Tensor<T> input_local = (input.ndims() == 1) ? input : input[i_batch];
    // class
    //   get tensors corresponding to target
    long istart = class_start_indices.at({cluster_target});
    Tensor<T> class_score_used, class_weight_used, class_bias_used;
    class_score_used.narrow(class_score[i_batch], 0, 0, cluster_size);
    class_weight_used.narrow(class_weight, 0, istart, cluster_size);
    class_bias_used.narrow(class_bias, 0, istart, cluster_size);
    //   compute score (input * weight + bias)
    class_score_used.addmv(1, class_bias_used, 1,
                           class_weight_used, input_local);
    assert(class_score_used.isContiguous());
    T* score_data = class_score_used.data();
    //   compute logsoftmax of score
    T maxInput = class_score_used.maxall();
    double class_logsum_local = 0.;
    for (int d = 0; d < cluster_size; ++d)
      class_logsum_local += THExpMinusApprox(maxInput - score_data[d]);
    class_logsum_local = maxInput + log(class_logsum_local);
    loss = class_logsum_local - class_score_used.at({idx_in_cluster_target});
    class_logsum.at({i_batch}) = class_logsum_local;
    // output
    output += loss;
  }
  // return value
  lua_pushnumber(L, output);
  lua_pushnumber(L, n_valid);
  return 2;
}

template <class T>
int updateGradInput(lua_State* L) {
  auto gradInput      = luaGetFieldIfTensorChecked<T>(L, 1, "gradInput");
  auto cluster_weight = luaGetFieldIfTensorChecked<T>(L, 1, "cluster_weight");
  auto class_weight   = luaGetFieldIfTensorChecked<T>(L, 1, "class_weight");
  auto class_score    = luaGetFieldIfTensorChecked<T>(L, 1, "class_score");
  auto class_logsum   = luaGetFieldIfTensorChecked<T>(L, 1, "class_logsum");
  auto mapping        = luaGetFieldIfTensorChecked<long>(L, 1, "mapping");
  auto n_class_in_cluster =
    luaGetFieldIfTensorChecked<long>(L, 1, "n_class_in_cluster");
  auto class_start_indices =
    luaGetFieldIfTensorChecked<long>(L, 1, "class_start_indices");
  auto target     = luaGetTensorChecked<long>(L, 2);
  auto n_clusters = cluster_weight.size(0);
  auto n_class    = class_weight.size(0);
  auto batch_size = gradInput.size(0);
  if (gradInput.ndims() == 1)
    batch_size = 1;
  for (int i_batch = 0; i_batch < batch_size; ++i_batch) {
    long itarget = target.at({i_batch}) - 1; // 1based->0based
    long cluster_target = mapping.at({itarget, 0}) - 1; // 1based->0based
    long idx_in_cluster_target = mapping.at({itarget, 1}) - 1; // 1based->0based
    long cluster_size = n_class_in_cluster.at({cluster_target});
    Tensor<T> gradInput_local =
      (gradInput.ndims() == 1) ? gradInput : gradInput[i_batch];
    // class:
    //   get tensors corresponding to target
    long istart = class_start_indices.at({cluster_target});
    Tensor<T> class_score_used, class_weight_used;
    class_score_used.narrow(class_score[i_batch], 0, 0, cluster_size);
    class_weight_used.narrow(class_weight, 0, istart, cluster_size);
    //   compute gradInput of the logsoftmax (into class_score)
    T class_logsum_local = class_logsum.at({i_batch});
    assert(class_score_used.isContiguous());
    T* score_data = class_score_used.data();
    for (int d = 0; d < cluster_size; ++d)
      score_data[d] = exp(score_data[d] - class_logsum_local);
    score_data[idx_in_cluster_target] -= 1.;
    //   compute gradInput of the addmv part
    Tensor<T> weight_t;
    weight_t.transpose(class_weight_used, 0, 1);
    gradInput_local.addmv(1, 1, weight_t, class_score_used);
  }

  return 0;
}

template <class T>
int accGradParameters(lua_State* L) {
  auto class_score    = luaGetFieldIfTensorChecked<T>(L, 1, "class_score");
  auto mapping        = luaGetFieldIfTensorChecked<long>(L, 1, "mapping");
  auto class_grad_weight   =
    luaGetFieldIfTensorChecked<T>(L, 1, "class_grad_weight");
  auto class_grad_bias     =
    luaGetFieldIfTensorChecked<T>(L, 1, "class_grad_bias");
  auto n_class_in_cluster =
    luaGetFieldIfTensorChecked<long>(L, 1, "n_class_in_cluster");
  auto class_start_indices =
    luaGetFieldIfTensorChecked<long>(L, 1, "class_start_indices");
  auto input      = luaGetTensorChecked<T>(L, 2);
  auto target     = luaGetTensorChecked<long>(L, 3);
  auto scale      = lua_tonumber(L, 4);
  auto batch_size = input.size(0);
  if (input.ndims() == 1)
    batch_size = 1;
  // class:
  for (int i_batch = 0; i_batch < batch_size; ++i_batch) {
    long itarget = target.at({i_batch}) - 1; // 1based->0based
    long cluster_target = mapping.at({itarget, 0}) - 1; // 1based->0based
    long idx_in_cluster_target = mapping.at({itarget, 1}) - 1; // 1based->0based
    long cluster_size = n_class_in_cluster.at({cluster_target});
    Tensor<T> input_local = (input.ndims() == 1) ? input : input[i_batch];
    //   get tensors corresponding to target
    long istart = class_start_indices.at({cluster_target});
    Tensor<T> class_score_used, class_grad_weight_used, class_grad_bias_used;
    class_score_used.narrow(class_score[i_batch], 0, 0, cluster_size);
    class_grad_weight_used.narrow(class_grad_weight, 0, istart, cluster_size);
    class_grad_bias_used.narrow(class_grad_bias, 0, istart, cluster_size);
    //   accumulate gradients
    class_grad_weight_used.addr(1, scale, class_score_used, input_local);
    class_grad_bias_used.cadd(scale, class_score_used);
  }
  return 0;
}

template <class T>
int accGradParameters_directUpdate(lua_State* L) {
  auto class_score    = luaGetFieldIfTensorChecked<T>(L, 1, "class_score");
  auto mapping        = luaGetFieldIfTensorChecked<long>(L, 1, "mapping");
  auto class_weight   = luaGetFieldIfTensorChecked<T>(L, 1, "class_weight");
  auto class_bias     = luaGetFieldIfTensorChecked<T>(L, 1, "class_bias");
  auto n_class_in_cluster =
    luaGetFieldIfTensorChecked<long>(L, 1, "n_class_in_cluster");
  auto class_start_indices =
    luaGetFieldIfTensorChecked<long>(L, 1, "class_start_indices");
  auto input      = luaGetTensorChecked<T>(L, 2);
  auto target     = luaGetTensorChecked<long>(L, 3);
  auto scale      = lua_tonumber(L, 4);
  auto batch_size = input.size(0);
  if (input.ndims() == 1)
    batch_size = 1;
  // class:
  for (int i_batch = 0; i_batch < batch_size; ++i_batch) {
    long itarget = target.at({i_batch}) - 1; // 1based->0based
    long cluster_target = mapping.at({itarget, 0}) - 1; // 1based->0based
    long idx_in_cluster_target = mapping.at({itarget, 1}) - 1; // 1based->0based
    long cluster_size = n_class_in_cluster.at({cluster_target});
    Tensor<T> input_local = (input.ndims() == 1) ? input : input[i_batch];
    //   get tensors corresponding to target
    long istart = class_start_indices.at({cluster_target});
    Tensor<T> class_score_used, class_weight_used, class_bias_used;
    class_score_used.narrow(class_score[i_batch], 0, 0, cluster_size);
    class_weight_used.narrow(class_weight, 0, istart, cluster_size);
    class_bias_used.narrow(class_bias, 0, istart, cluster_size);
    //   accumulate gradients
    class_weight_used.addr(1, scale, class_score_used, input_local);
    class_bias_used.cadd(scale, class_score_used);
  }
  return 0;
}

template <class T>
int zeroGradParametersClass(lua_State* L) {
  auto mapping        = luaGetFieldIfTensorChecked<long>(L, 1, "mapping");
  auto class_grad_weight   = luaGetFieldIfTensorChecked<T>(L, 1,
                                                           "class_grad_weight");
  auto class_grad_bias     = luaGetFieldIfTensorChecked<T>(L, 1,
                                                           "class_grad_bias");
  auto n_class_in_cluster =
    luaGetFieldIfTensorChecked<long>(L, 1, "n_class_in_cluster");
  auto class_start_indices =
    luaGetFieldIfTensorChecked<long>(L, 1, "class_start_indices");
  auto target     = luaGetTensorChecked<long>(L, 2);
  auto batch_size = target.size(0);
  // TODO: be smarter and 0 out only once per cluster.
  for (int i_batch = 0; i_batch < batch_size; ++i_batch) {
    long itarget = target.at({i_batch}) - 1; // 1based->0based
    long cluster_target = mapping.at({itarget, 0}) - 1; // 1based->0based
    long idx_in_cluster_target = mapping.at({itarget, 1}) - 1; // 1based->0based
    long cluster_size = n_class_in_cluster.at({cluster_target});
    //   get tensors corresponding to target
    long istart = class_start_indices.at({cluster_target});
    Tensor<T> class_grad_weight_used, class_grad_bias_used;
    class_grad_weight_used.narrow(class_grad_weight, 0, istart, cluster_size);
    class_grad_bias_used.narrow(class_grad_bias, 0, istart, cluster_size);
    //   accumulate gradients
    class_grad_weight_used.fill(0.0);
    class_grad_bias_used.fill(0.0);
  }
  return 0;
}

template <class T>
class Registerer {
 private:
  static const luaL_Reg functions_[];
public:
  static void registerFunctions(lua_State* L);
};

template <class T>
const luaL_Reg Registerer<T>::functions_[] = {
  {"HSM_updateOutputWithTarget"        , updateOutputWithTarget<T>},
  {"HSM_updateGradInput"               , updateGradInput<T>},
  {"HSM_accGradParameters"             , accGradParameters<T>},
  {"HSM_accGradParameters_directUpdate", accGradParameters_directUpdate<T>},
  {"HSM_zeroGradParametersClass"       , zeroGradParametersClass<T>},
  {nullptr, nullptr},
};

template <class T>
void Registerer<T>::registerFunctions(lua_State* L) {
  luaT_pushmetatable(L, Tensor<T>::kLuaTypeName);
  luaT_registeratname(L, functions_, "nn");
  lua_pop(L, 1);
}

} // namespace

void initHSM(lua_State* L) {
  Registerer<float>::registerFunctions(L);
  Registerer<double>::registerFunctions(L);
}

}}} // namespaces

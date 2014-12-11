/**
 * Copyright 2014 Facebook
 * @author Misha Denil (mdenil@fb.com)
 */

#include <lua.hpp>
#include <luaT.h>

#include <vector>
#include <queue>
#include <utility>
#include <stdexcept>

// Turn off range checking for multi_array indexing
#define BOOST_DISABLE_ASSERTS
#include <boost/multi_array.hpp>

#include "torch/fb/fbcunn/layers/LuaUtils.h"
#include "torch/fb/fbcunn/layers/Tensor.h"

namespace facebook { namespace deeplearning { namespace torch {

namespace {

template<typename T>
struct index_value : public std::pair<long, T> {
  index_value(long const& i, T const& v): std::pair<long, T>(i, v) {}
  long const& index() const { return this->first; }
  T const& value() const { return this->second; }
};

template<typename T>
inline index_value<T> make_index_value(long const& i, T const& v) {
  return index_value<T>(i, v);
}

template<typename T>
struct value_order {
  bool operator()(index_value<T> const& a, index_value<T> const& b) {
    return a.value() > b.value();
  }
};

template<typename T>
struct index_order {
  bool operator()(index_value<T> const& a, index_value<T> const& b) {
    return a.index() < b.index();
  }
};

template<typename T>
boost::multi_array_ref<T, 2> tensor_array_view(Tensor<T>& tensor) {
  return boost::multi_array_ref<T, 2>(
    tensor.data(),
    boost::extents[tensor.size(0)][tensor.size(1)]);
}

template<typename T>
boost::const_multi_array_ref<T, 2> tensor_array_view(Tensor<T> const& tensor) {
  return boost::const_multi_array_ref<T, 2>(
    tensor.data(),
    boost::extents[tensor.size(0)][tensor.size(1)]);
}

template<typename T>
boost::multi_array_ref<T, 1> tensor_vector_view(Tensor<T>& tensor) {
  return boost::multi_array_ref<T, 1>(
    tensor.data(),
    boost::extents[tensor.size(0)]);
}

template<typename T>
boost::const_multi_array_ref<T, 1> tensor_vector_view(Tensor<T> const& tensor) {
  return boost::const_multi_array_ref<T, 1>(
    tensor.data(),
    boost::extents[tensor.size(0)]);
}

template<typename T>
using value_priority_queue = std::priority_queue<
  index_value<T>,
  std::vector<index_value<T>>,
  value_order<T>>;

// Forward pass for a single example
template<typename T>
void updateOutput_single(
  Tensor<T> const& input_tensor,
  Tensor<T> const& norms_tensor,
  Tensor<long>& switches_tensor,
  Tensor<T>& output_tensor,
  long k) {

  auto input_data = tensor_array_view(input_tensor);
  auto norms_data = tensor_vector_view(norms_tensor);
  auto output_data = tensor_array_view(output_tensor);
  auto switches_data = tensor_vector_view(switches_tensor);

  // compute the k-max norms
  value_priority_queue<T> kmax;
  for (int row = 0; row < norms_tensor.size(0); ++row) {
    T value = norms_data[row];

    if (kmax.size() < k || value > kmax.top().value()) {
      kmax.push(make_index_value(row, value));

      if (kmax.size() > k) {
        kmax.pop();
      }
    }
  }

  // re-order k-max into input order
  std::vector<index_value<T>> kmax_sorted;
  while (!kmax.empty()) {
    kmax_sorted.push_back(kmax.top());
    kmax.pop();
  }
  std::sort(
    kmax_sorted.begin(),
    kmax_sorted.end(),
    index_order<T>());

  // copy the k-max indices into switches
  std::transform(
    kmax_sorted.begin(),
    kmax_sorted.end(),
    switches_data.begin(),
    [](index_value<T> const& v) -> long {
      return v.index();
    });

  // copy each k-max row to the output
  for (int row = 0; row < kmax_sorted.size(); ++row) {
    long input_row = switches_data[row];

    for (int col = 0; col < input_tensor.size(1); ++col) {
      output_data[row][col] = input_data[input_row][col];
    }
  }

}

// Forward pass
template <class T>
int updateOutput(lua_State* L) {
  auto const input_tensor = luaGetTensorChecked<T>(L, 2);
  auto const norms_tensor = luaGetTensorChecked<T>(L, 3);
  auto switches_tensor = luaGetFieldIfTensorChecked<long>(L, 1, "switches");
  auto k = luaGetFieldIfNumberChecked<long>(L, 1, "k");
  auto k_dynamic = luaGetFieldIfNumberChecked<double>(L, 1, "k_dynamic");
  auto output_tensor = luaGetFieldIfTensorChecked<T>(L, 1, "output");
  int outputIdx = lua_gettop(L);

  luaL_argcheck(L,
    input_tensor.ndims() == 2 || input_tensor.ndims() == 3,
    1,
    "input must have 2 or 3 dimensions.");


  // Set k for dynamic k-max pooling
  if (k_dynamic > 0) {
    int sentence_dim = input_tensor.ndims() - 2;
    k = std::max(k, static_cast<long>(k_dynamic * input_tensor.size(sentence_dim)));
  }

  if (input_tensor.ndims() == 2) {
    output_tensor.resize(LongStorage{k, input_tensor.size(1)});
    switches_tensor.resize(LongStorage{k});
    output_tensor.fill(0.0);
    switches_tensor.fill(0);

    updateOutput_single(
      input_tensor,
      norms_tensor,
      switches_tensor,
      output_tensor,
      k);
  }
  else {
    output_tensor.resize(
      LongStorage{input_tensor.size(0), k, input_tensor.size(2)});
    switches_tensor.resize(
      LongStorage{input_tensor.size(0), k});
    output_tensor.fill(0.0);
    switches_tensor.fill(0);

    #pragma omp parallel for
    for (int i = 0; i < input_tensor.size(0); ++i) {
      // Copy meta-data but not real-data
      Tensor<T> input_tensor_example = input_tensor;
      Tensor<T> norms_tensor_example = norms_tensor;
      Tensor<long> switches_tensor_example = switches_tensor;
      Tensor<T> output_tensor_example = output_tensor;

      // Narrow to the current example
      input_tensor_example.select(0, i);
      norms_tensor_example.select(0, i);
      switches_tensor_example.select(0, i);
      output_tensor_example.select(0, i);

      updateOutput_single(
        input_tensor_example,
        norms_tensor_example,
        switches_tensor_example,
        output_tensor_example,
        k);
    }
  }

  lua_pushvalue(L, outputIdx);
  return 1;
}

// Backprop for a single example
template<typename T>
void updateGradInput_singe(
  Tensor<T> const& gradOutput_tensor,
  Tensor<long> const& switches_tensor,
  Tensor<T>& gradInput_tensor) {

  auto gradInput_data = tensor_array_view(gradInput_tensor);
  auto const gradOutput_data = tensor_array_view(gradOutput_tensor);
  auto const switches_data = tensor_vector_view(switches_tensor);

  for (int row = 0; row < gradOutput_tensor.size(0); ++row) {
    long input_row = switches_data[row];

    for (int col = 0; col < gradOutput_tensor.size(1); ++col) {
      gradInput_data[input_row][col] = gradOutput_data[row][col];
    }
  }
}

// Backprop
template <class T>
int updateGradInput(lua_State* L) {
  auto const input_tensor = luaGetTensorChecked<T>(L, 2);
  auto const gradOutput_tensor = luaGetTensorChecked<T>(L, 3);
  auto const switches_tensor =
    luaGetFieldIfTensorChecked<long>(L, 1, "switches");
  auto gradInput_tensor = luaGetFieldIfTensorChecked<T>(L, 1, "gradInput");
  int gradInputIdx = lua_gettop(L);

  luaL_argcheck(L,
    gradOutput_tensor.ndims() == 2 || gradOutput_tensor.ndims() == 3,
    1,
    "gradOutput must have exactly 2 or 3 dimensions.");

  gradInput_tensor.resizeAs(input_tensor);
  gradInput_tensor.fill(0.0);

  if (gradOutput_tensor.ndims() == 2) {
    updateGradInput_singe(
      gradOutput_tensor,
      switches_tensor,
      gradInput_tensor);
  }
  else {

    #pragma omp parallel for
    for (int i = 0; i < gradOutput_tensor.size(0); ++i) {
      // Copy meta-data but not real-data
      Tensor<T> gradOutput_tensor_example = gradOutput_tensor;
      Tensor<long> switches_tensor_example = switches_tensor;
      Tensor<T> gradInput_tensor_example = gradInput_tensor;

      // Narrow to the current example
      gradOutput_tensor_example.select(0, i);
      switches_tensor_example.select(0, i);
      gradInput_tensor_example.select(0, i);

      updateGradInput_singe(
        gradOutput_tensor_example,
        switches_tensor_example,
        gradInput_tensor_example);
    }
  }

  lua_pushvalue(L, gradInputIdx);
  return 1;
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
  {"GroupKMaxPooling_updateOutput", updateOutput<T>},
  {"GroupKMaxPooling_updateGradInput", updateGradInput<T>},
  {nullptr, nullptr},
};

template <class T>
void Registerer<T>::registerFunctions(lua_State* L) {
  luaT_pushmetatable(L, Tensor<T>::kLuaTypeName);
  luaT_registeratname(L, functions_, "nn");
  lua_pop(L, 1);
}

}  // namespace

void initGroupKMaxPooling(lua_State* L) {
  Registerer<float>::registerFunctions(L);
  Registerer<double>::registerFunctions(L);
}

}}}  // namespaces

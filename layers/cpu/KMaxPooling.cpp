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
using array_view = boost::multi_array_ref<T, 2>;

template<typename T>
using array_col_view = typename array_view<T>::template array_view<1>::type;

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

// Forward pass
template <class T>
int updateOutput(lua_State* L) {
  auto const input_tensor = luaGetTensorChecked<T>(L, 2);
  auto input_length_tensor
    = luaGetFieldIfTensorChecked<long>(L, 1, "input_length");
  auto output_length_tensor
    = luaGetFieldIfTensorChecked<long>(L, 1, "output_length");
  auto switches_tensor = luaGetFieldIfTensorChecked<long>(L, 1, "switches");
  auto k = luaGetFieldIfNumberChecked<long>(L, 1, "k");
  auto k_dynamic = luaGetFieldIfNumberChecked<double>(L, 1, "k_dynamic");
  auto output_tensor = luaGetFieldIfTensorChecked<T>(L, 1, "output");
  int outputIdx = lua_gettop(L);

  luaL_argcheck(L, input_tensor.ndims() == 2, 1,
    "input must have exactly 2 dimensions.");
  luaL_argcheck(L, input_length_tensor.ndims() == 1, 1,
    "input_length must have exactly 1 dimension");
  luaL_argcheck(L, input_length_tensor.size() == 1, 1,
    "input_length must have exactly 1 element");

  // Set k for dynamic k-max pooling
  if (k_dynamic > 0) {
    int sentence_dim = input_tensor.ndims() - 2;
    k = std::max(k, static_cast<long>(k_dynamic * input_tensor.size(sentence_dim)));
  }

  output_tensor.resize(LongStorage{k, input_tensor.size(1)});
  switches_tensor.resize(LongStorage{k, input_tensor.size(1)});
  output_tensor.fill(0.0);
  switches_tensor.fill(0);

  auto input_data = tensor_array_view(input_tensor);
  auto output_data = tensor_array_view(output_tensor);
  auto switches_data = tensor_array_view(switches_tensor);
  auto input_length_data = tensor_vector_view(input_length_tensor);
  auto output_length_data = tensor_vector_view(output_length_tensor);

  for (int col = 0; col < input_tensor.size(1); ++col) {
    value_priority_queue<T> kmax;

    // extract the k-max values with indices
    // for (int row = 0; row < input_tensor.size(0); ++row) {
    for (int row = 0; row < input_length_data[0]; ++row) {
      T value = input_data[row][col];

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

    // copy the k-max values to the output
    array_col_view<T> output_col = output_data[
      boost::indices[typename array_view<T>::index_range()][col]];
    std::transform(
      kmax_sorted.begin(),
      kmax_sorted.end(),
      output_col.begin(),
      [](index_value<T> const& v) -> T {
        return v.value();
      });

    // copy the k-max indices into switches
    array_col_view<long> switches_col = switches_data[
      boost::indices[typename array_view<long>::index_range()][col]];
    std::transform(
      kmax_sorted.begin(),
      kmax_sorted.end(),
      switches_col.begin(),
      [](index_value<T> const& v) -> long {
        return v.index();
      });
  }

  // update length after pooling
  output_length_data[0] = std::min(input_length_data[0], k);

  lua_pushvalue(L, outputIdx);
  return 1;
}

// Backprop
template <class T>
int updateGradInput(lua_State* L) {
  auto const input = luaGetTensorChecked<T>(L, 2);
  auto const gradOutput_tensor = luaGetTensorChecked<T>(L, 3);
  auto input_length_tensor
    = luaGetFieldIfTensorChecked<long>(L, 1, "input_length");
  auto output_length_tensor
    = luaGetFieldIfTensorChecked<long>(L, 1, "output_length");
  auto const switches_tensor =
    luaGetFieldIfTensorChecked<long>(L, 1, "switches");
  auto gradInput_tensor = luaGetFieldIfTensorChecked<T>(L, 1, "gradInput");
  int gradInputIdx = lua_gettop(L);

  luaL_argcheck(L, gradOutput_tensor.ndims() == 2, 1,
    "gradOutput must have exactly 2 dimensions.");
  luaL_argcheck(L, input_length_tensor.ndims() == 1, 1,
    "input_length must have exactly 1 dimension");
  luaL_argcheck(L, input_length_tensor.size() == 1, 1,
    "input_length must have exactly 1 element");

  long output_length = output_length_tensor.front();

  gradInput_tensor.resizeAs(input);
  gradInput_tensor.fill(0.0);

  auto gradInput_data = tensor_array_view(gradInput_tensor);
  auto const gradOutput_data = tensor_array_view(gradOutput_tensor);
  auto const switches_data = tensor_array_view(switches_tensor);

  long row_limit = std::min(output_length, gradOutput_tensor.size(0));
  for (int row = 0; row < row_limit; ++row) {
    for (int col = 0; col < gradOutput_tensor.size(1); ++col) {
      long input_row = switches_data[row][col];
      gradInput_data[input_row][col] = gradOutput_data[row][col];
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
  {"KMaxPooling_updateOutput", updateOutput<T>},
  {"KMaxPooling_updateGradInput", updateGradInput<T>},
  {nullptr, nullptr},
};

template <class T>
void Registerer<T>::registerFunctions(lua_State* L) {
  luaT_pushmetatable(L, Tensor<T>::kLuaTypeName);
  luaT_registeratname(L, functions_, "nn");
  lua_pop(L, 1);
}

}  // namespace

void initKMaxPooling(lua_State* L) {
  Registerer<float>::registerFunctions(L);
  Registerer<double>::registerFunctions(L);
}

}}}  // namespaces

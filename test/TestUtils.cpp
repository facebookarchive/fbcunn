// Copyright 2004-present Facebook. All Rights Reserved.

#include "TestUtils.h"
#include "THCTensor.h"

#include <folly/Random.h>
#include <glog/logging.h>
#include <iomanip>
#include <string>
#include <sstream>
#include <utility>

using namespace std;
using namespace thpp;
using namespace facebook::deeplearning::torch;

namespace facebook { namespace deeplearning { namespace torch { namespace test {
// Constructs a CUDA tensor of the same size as the input
unique_ptr<THCudaTensor, CudaTensorDeleter>
makeTHCudaTensorSameSize(THCState* state, const Tensor<float>& t) {
  vector<long> sizes;
  vector<long> strides;
  for (int i = 0; i < t.ndims(); ++i) {
    sizes.push_back(t.size(i));
    strides.push_back(t.stride(i));
  }

  return makeTHCudaTensorFull(state, sizes, strides);
}


Tensor<float>
makeTestTensor(
  initializer_list<long> sizeList,
  initializer_list<float> factorList,
  const std::function<float(long, long, long, long)>& computeVals,
  const folly::Optional<tuple<long, long, long, long>>& padding = folly::none) {
  CHECK_EQ(4, sizeList.size());

  const auto sizes = vector<long>{sizeList.begin(), sizeList.end()};
  auto sizesWithPadding = sizes;

  const auto topPad = padding ? get<0>(*padding) : 0L;
  const auto bottomPad = padding ? get<1>(*padding) : 0L;
  const auto leftPad = padding ? get<2>(*padding) : 0L;
  const auto rightPad = padding ? get<3>(*padding) : 0L;
  sizesWithPadding[2] += topPad + bottomPad;
  sizesWithPadding[3] += leftPad + rightPad;

  auto out = Tensor<float>{sizesWithPadding};

  for (long l = 0; l < sizesWithPadding[0]; ++l) {
    for (long k = 0; k < sizesWithPadding[1]; ++k) {
      for (long j = 0; j < sizesWithPadding[2]; ++j) {
        for (long i = 0; i < sizesWithPadding[3]; ++i) {
          if (j < topPad || j >= sizes[2] + topPad ||
              i < leftPad || i >= sizes[3] + leftPad) {
            out.at({l, k, j, i}) = 0.0f;
          } else {
            out.at({l, k, j, i}) = computeVals(i - leftPad, j - topPad, k, l);
          }
        }
      }
    }
  }

  return out;
}

// Populate with random
Tensor<float>
makeRandomTestTensor(initializer_list<long> sizeList) {
  return makeTestTensor(sizeList,
                        {0.0f, 0.0f, 0.0f, 0.0f},
                        [](long i, long j, long k, long l) {
                          return (float)(folly::Random::rand32(1000)) / 1000.0f;
                        });
}

// Populate with constant
Tensor<float>
makeTestTensor(initializer_list<long> sizeList, float constant) {
  return makeTestTensor(sizeList,
                        {0.0f, 0.0f, 0.0f, 0.0f},
                        [constant](long i, long j, long k, long l) {
                          return constant;
                        });
}

// Populate with factorList
Tensor<float>
makeTestTensor(initializer_list<long> sizeList,
               initializer_list<float> factorList,
               const folly::Optional<tuple<long, long, long, long>>& padding) {

  CHECK_EQ(sizeList.size(), factorList.size());
  const auto factors = vector<float>{factorList.begin(), factorList.end()};
  return makeTestTensor(sizeList, factorList,
                        [factors](long i, long j, long k, long l) {
                          return factors[0] * l + factors[1] * k +
                            factors[2] * j + factors[3] * i;
                        },
                        padding);
}

// Populate with {0.1f, 0.2f, 0.3f, 0.4f}
Tensor<float>
makeTestTensor(initializer_list<long> sizeList) {
  return makeTestTensor(sizeList,
                        {0.1f, 0.2f, 0.3f, 0.4f});
}


//
// Utilities for debugging tensor differences
//

bool isWithin(float a, float b, float relativeError) {
  const auto adjRelativeError = 0.5f * relativeError;

  // Handle special cases
  if (a == b || (std::isnan(a) && std::isnan(b))) {
    return true;
  } else if (!std::isfinite(a) && !std::isfinite(b)) {
    if (std::signbit(a) == std::signbit(b)) {
      return true;
    } else {
      return false;
    }
  }


  // Special case for a or b very close to zero, only absolute check can work
  if (std::abs(a) < relativeError || std::abs(a) < relativeError ||
      !std::isnormal(a)|| !std::isnormal(b)) {
    if (std::abs(a - b) > adjRelativeError) {
      return false;
    }
    return true;
  }

  // Compare the difference against the mean values
  if (std::abs(a - b) > adjRelativeError * (std::abs(a) + std::abs(b))) {
    return false;
  }

  return true;
}

long dimUB(const Tensor<float>& reference,
                   const Tensor<float>& test,
                   int dim,
                   bool compareCommonPartOnly) {
  return compareCommonPartOnly ?
    std::min(reference.size(dim), test.size(dim)) :
    test.size(dim);
}

std::pair<bool, std::string>
compareTensors1d(const Tensor<float>& reference,
                 const Tensor<float>& test,
                 float relativeError,
                 int precisionDebug) {
  CHECK_EQ(1, reference.ndims());
  CHECK_EQ(reference.ndims(), test.ndims());
  for (int i = 0; i < reference.ndims(); ++i) {
    CHECK_EQ(reference.size(i), test.size(i));
  }

  std::ostringstream ss;

  bool error = false;
  for (int i = 0; i < reference.size(0); ++i) {
    auto refVal = reference.at({i});
    auto testVal = test.at({i});

    if (!isWithin(refVal, testVal, relativeError)) {
      error = true;
      break;
    }
  }

  if (!error) {
    return make_pair(true, std::string());
  }

  ss << "Mismatch" << std::endl;
  ss << std::setprecision(precisionDebug) << std::fixed;

  for (int i = 0; i < reference.size(0); ++i) {
    ss << std::setw(8) << reference.at({i}) << " ";
  }

  ss << "| ";

  for (int i = 0; i < reference.size(0); ++i) {
    if (!isWithin(reference.at({i}), test.at({i}), relativeError)) {
      ss << "*" << std::setw(7);
    } else {
      ss << std::setw(8);
    }

    ss << test.at({i}) << " ";
  }

  ss << std::endl;
  return make_pair(false, ss.str());
}

std::pair<bool, std::string>
compareTensors4d(const Tensor<float>& reference,
                 const Tensor<float>& test,
                 float relativeError,
                 int precisionDebug,
                 bool compareInter) {

  CHECK_EQ(4, reference.ndims());
  if (!compareInter) {
    CHECK_EQ(reference.ndims(), test.ndims());
    for (int i = 0; i < reference.ndims(); ++i) {
      CHECK_EQ(reference.size(i), test.size(i));
    }
  }

  std::ostringstream ss;

  for (int l = 0; l < dimUB(reference, test, 0, compareInter); ++l) {
    for (int k = 0; k < dimUB(reference, test, 1, compareInter); ++k) {

      bool error = false;
      for (int j = 0; j < dimUB(reference, test, 2, compareInter); ++j) {
        for (int i = 0; i < dimUB(reference, test, 3, compareInter); ++i) {
          auto refVal = reference.at({l, k, j, i});
          auto testVal = test.at({l, k, j, i});

          if (!isWithin(refVal, testVal, relativeError)) {
            error = true;
            break;
          }
        }

        if (error) {
          // We'll produce the first 2d slice on which there is a
          // difference below
          break;
        }
      }

      if (!error) {
        continue;
      }

      ss << "Mismatch on (" << l << ", " << k << ")" << std::endl;
      ss << std::setprecision(precisionDebug) << std::fixed;

      for (int j = 0; j < dimUB(reference, test, 2, compareInter); ++j) {
        for (int i = 0; i < dimUB(reference, test, 3, compareInter); ++i) {
          ss << std::setw(8) << reference[l][k][j][i].front() << " ";
        }

        ss << "| ";

        for (int i = 0; i < dimUB(reference, test, 3, compareInter); ++i) {
          if (!isWithin(reference[l][k][j][i].front(),
                        test[l][k][j][i].front(),
                        relativeError)) {
            ss << "*" << std::setw(7);
          } else {
            ss << std::setw(8);
          }

          ss << test[l][k][j][i].front() << " ";
        }

        ss << std::endl;
      }

      // Return the first matrix mismatch
      return make_pair(false, ss.str());
    }
  }

  return make_pair(true, std::string());
}


std::pair<bool, std::string>
compareTensors3d(const Tensor<float>& reference,
                 const Tensor<float>& test,
                 float relativeError,
                 int precisionDebug,
                 bool compareInter) {

  auto reference4d = Tensor<float>();
  reference4d.resizeAs(reference);
  reference4d.copy(reference);
  reference4d.resize(
    {reference.size(0), reference.size(1), reference.size(2), 1}
  );

  auto test4d = Tensor<float>();
  test4d.resizeAs(test);
  test4d.copy(test);
  test4d.resize({test.size(0), test.size(1), test.size(2), 1});

  return compareTensors4d(reference4d, test4d, relativeError, precisionDebug,
                          compareInter);
}

// Returns true or false if the two tensors match within some relative
// error; also returns the 2d slice where they first differ as a
// string if they do.
std::pair<bool, std::string>
compareTensors(const Tensor<float>& reference,
               const Tensor<float>& test,
               float relativeError,
               int precisionDebug,
               bool compareInter) {
  CHECK_EQ(reference.ndims(), test.ndims());

  switch (reference.ndims()) {
    case 1:
      return compareTensors1d(reference, test, relativeError, precisionDebug);
    case 3:
      return compareTensors3d(reference, test, relativeError, precisionDebug,
                              compareInter);
    case 4:
      return compareTensors4d(reference, test, relativeError, precisionDebug,
                              compareInter);
    default:
      // unimplemented
      CHECK(false) << "unimplemented";
  }

  return make_pair(false, string());
}

}}}} // namespace

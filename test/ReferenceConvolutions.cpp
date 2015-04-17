// Copyright 2004-present Facebook. All Rights Reserved.

#include "torch/fb/fbcunn/test/ReferenceConvolutions.h"

#include <glog/logging.h>

using namespace std;

namespace facebook { namespace deeplearning { namespace torch { namespace test {

// How padding works
//
// Padding is implied space of zeros around the input. The output is
// enlarged by the contribution of the zero space. In other words, if
// padding = filter size - 1, the output would effectively be a full
// convolution. Padding is typically something a lot smaller though.
//
// For going in the reverse direction (output * filter => input in a
// full convolution), because the padded space is implied, there is
// nothing to update in the padded area, so the full convolution of
// output * filter operates with a mask when going to the input.
//
// -------------------------------
// |      implied zeros          |\
// |  _________________________  | \
// |  |                       |  |  \  convoled with
// |  |       real input      |  |   \____
// |  |                       |  |    |  |
// |  |                       |  | *  |  |  equals ==>
// |  |         area          |  |    ----
// |  |                       |  |   /
// |  |                       |  |  /
// |  -------------------------  | /
// |       implied zeros         |/
// -------------------------------
//
//         total output area
// -------------------------------
// |  affected by implied zeros  |
// |  _________________________  |
// |  |                       |  |
// |  |   output area not     |  |
// |  |     affected by       |  |
// |  |    implied zero       |  |
// |  |    area; this data    |  |
// |  |    is equivalent to   |  |
// |  |    pad (0, 0, 0, 0)   |  |
// |  -------------------------  |
// |  affected by implied zeros  |
// -------------------------------
//
// but for backprop convolution, the input area with implied zeros
// functions as a mask, so that we effectively ignore anything that
// would have come from the partial
//
//         total output area
// -------------------------------
// |  affected by implied zeros  |\
// |  _________________________  | \  convolved with
// |  |                       |  |  \
// |  |   output area not     |  |   \____
// |  |     affected by       |  |    |  |
// |  |    implied zero       |  | *  |  |  equals ==>
// |  |    area; this data    |  |    ----
// |  |    is equivalent to   |  |   /
// |  |    pad (0, 0, 0, 0)   |  |  /
// |  -------------------------  | /
// |  affected by implied zeros  |/
// -------------------------------
//
// -------------------------------
// |     calculations ignored    | (area with implied zeros contains
// |  _________________________  |  data from the output area
// |  |                       |  |  affected by implied zeros)
// |  |       real input      |  |
// |  |                       |  |
// |  |                       |  |
// |  |         area          |  |
// |  |                       |  |
// |  |                       |  |
// |  -------------------------  |
// |     calculations ignored    |
// -------------------------------

Tensor<float>
crossCorrelationValidOnly(
  const Tensor<float>& input,
  const Tensor<float>& filters,
  long filterRowStride,
  long filterColStride,
  const folly::Optional<std::tuple<long, long, long, long>>& padding) {

  const long inputPaddingTop = padding ? get<0>(*padding) : 0;
  const long inputPaddingBottom = padding ? get<1>(*padding) : 0;
  const long inputPaddingLeft = padding ? get<2>(*padding) : 0;
  const long inputPaddingRight = padding ? get<3>(*padding) : 0;

  if (input.ndims() != 4 || filters.ndims() != 4) {
    throw std::invalid_argument("illegal input/filter format");
  }

  if (input.size(1) != filters.size(1)) {
    throw std::invalid_argument("num planes mismatch");
  }

  if (filterRowStride < 1 || filterColStride < 1) {
    throw std::invalid_argument("filter stride should be >= 1");
  }

  if (filters.size(2) > input.size(2)) {
    throw std::invalid_argument("filter rows must be <= input rows in size");
  }

  if (filters.size(3) > input.size(3)) {
    throw std::invalid_argument("filter cols must be <= input cols in size");
  }

  // The filter must fit in an integral number of times into the image
  // after the stride
  if ((input.size(2) - filters.size(2)) % filterRowStride != 0) {
    throw std::invalid_argument("illegal row stride");
  }

  if ((input.size(3) - filters.size(3)) % filterColStride != 0) {
    throw std::invalid_argument("illegal col stride");
  }

  const auto outputRows =
    getValidConvSize(input.size(2) + inputPaddingTop + inputPaddingBottom,
                     filters.size(2), filterRowStride);
  const auto outputCols =
    getValidConvSize(input.size(3) + inputPaddingLeft + inputPaddingRight,
                     filters.size(3), filterColStride);

  auto output =
    Tensor<float>{{input.size(0), filters.size(0), outputRows, outputCols}};
  output.fill(0);

  // doall
  for (long batch = 0; batch < input.size(0); ++batch) {
    // doall
    for (long filter = 0; filter < filters.size(0); ++filter) {
      // reduct
      for (long plane = 0; plane < input.size(1); ++plane) {
        // doall
        for (long outputRow = 0; outputRow < outputRows; ++outputRow) {
          const auto inputRowStartAfterPadding =
            outputRow * filterRowStride;
          const auto inputRowStartBeforePadding =
            inputRowStartAfterPadding - inputPaddingTop;

          // doall
          for (long outputCol = 0; outputCol < outputCols; ++outputCol) {
            const auto inputColStartAfterPadding =
              outputCol * filterColStride;
            const auto inputColStartBeforePadding =
              inputColStartAfterPadding - inputPaddingLeft;

            float val = 0.0f;

            // reduct
            for (long filterRow = 0; filterRow < filters.size(2); ++filterRow) {
              const auto inputRowBeforePadding =
                inputRowStartBeforePadding + filterRow;

              if (inputRowBeforePadding < 0 ||
                  inputRowBeforePadding >= input.size(2)) {
                continue;
              }

              // reduct
              for (long filterCol = 0; filterCol < filters.size(3);
                   ++filterCol) {
                const auto inputColBeforePadding =
                  inputColStartBeforePadding + filterCol;

                if (inputColBeforePadding < 0 ||
                    inputColBeforePadding >= input.size(3)) {
                  continue;
                }

                const auto in =
                  input.at({batch, plane,
                            inputRowBeforePadding, inputColBeforePadding});
                const auto filt =
                  filters.at({filter, plane, filterRow, filterCol});

                val += filt * in;
              }
            }

            output.at({batch, filter, outputRow, outputCol}) += val;
          }
        }
      }
    }
  }

  return output;
}

Tensor<float>
crossCorrelationValidOnlyInputCentric(
  const Tensor<float>& input,
  const Tensor<float>& filters,
  long filterRowStride,
  long filterColStride,
  const folly::Optional<std::tuple<long, long, long, long>>& padding) {

  const long inputPaddingTop = padding ? get<0>(*padding) : 0;
  const long inputPaddingBottom = padding ? get<1>(*padding) : 0;
  const long inputPaddingLeft = padding ? get<2>(*padding) : 0;
  const long inputPaddingRight = padding ? get<3>(*padding) : 0;

  if (input.ndims() != 4 || filters.ndims() != 4) {
    throw std::invalid_argument("illegal input/filter format");
  }

  if (input.size(1) != filters.size(1)) {
    throw std::invalid_argument("num planes mismatch");
  }

  if (filterRowStride < 1 || filterColStride < 1) {
    throw std::invalid_argument("filter stride should be >= 1");
  }

  if (filters.size(2) > input.size(2)) {
    throw std::invalid_argument("filter rows must be <= input rows in size");
  }

  if (filters.size(3) > input.size(3)) {
    throw std::invalid_argument("filter cols must be <= input cols in size");
  }

  // The filter must fit in an integral number of times into the image
  // after the stride
  if ((input.size(2) - filters.size(2)) % filterRowStride != 0) {
    throw std::invalid_argument("illegal row stride");
  }

  if ((input.size(3) - filters.size(3)) % filterColStride != 0) {
    throw std::invalid_argument("illegal col stride");
  }

  const auto inputRowsBeforePadding = input.size(2);
  const auto inputColsBeforePadding = input.size(3);

  const auto outputRows =
    getValidConvSize(inputRowsBeforePadding +
                     inputPaddingTop + inputPaddingBottom,
                     filters.size(2), filterRowStride);
  const auto outputCols =
    getValidConvSize(inputColsBeforePadding +
                     inputPaddingLeft + inputPaddingRight,
                     filters.size(3), filterColStride);

  auto output =
    Tensor<float>{{input.size(0), filters.size(0), outputRows, outputCols}};
  output.fill(0);

  int count = 0;

  // doall
  for (long batch = 0; batch < input.size(0); ++batch) {
    // doall
    for (long filter = 0; filter < filters.size(0); ++filter) {
      // reduct
      for (long plane = 0; plane < input.size(1); ++plane) {
        // doall
        for (long inputRowBeforePadding = 0;
             inputRowBeforePadding < inputRowsBeforePadding;
             ++inputRowBeforePadding) {
          const auto inputRowAfterPadding =
            inputRowBeforePadding + inputPaddingTop;

          // doall
          for (long inputColBeforePadding = 0;
               inputColBeforePadding < inputColsBeforePadding;
               ++inputColBeforePadding) {
            const auto inputColAfterPadding =
              inputColBeforePadding + inputPaddingLeft;

            const auto in =
              input.at({batch, plane,
                        inputRowBeforePadding, inputColBeforePadding});

            for (long filterRow = inputRowAfterPadding % filterRowStride;
                 filterRow <= std::min(filters.size(2) - 1,
                                       inputRowAfterPadding);
                 filterRow += filterRowStride) {
              CHECK_GE(inputRowAfterPadding, filterRow);
              CHECK_EQ(0, (inputRowAfterPadding - filterRow) % filterRowStride);

              const auto outputRow =
                (inputRowAfterPadding - filterRow) / filterRowStride;
              if (outputRow >= outputRows) {
                continue;
              }

              for (long filterCol = inputColAfterPadding % filterColStride;
                   filterCol <= std::min(filters.size(3) - 1,
                                         inputColAfterPadding);
                   filterCol += filterColStride) {
                CHECK_GE(inputColAfterPadding, filterCol);
                CHECK_EQ(0, (inputColAfterPadding - filterCol) %
                         filterColStride);

                const auto outputCol =
                  (inputColAfterPadding - filterCol) / filterColStride;
                if (outputCol >= outputCols) {
                  continue;
                }

                CHECK_LE(0, filter);
                CHECK_LE(0, plane);
                CHECK_LE(0, filterRow);
                CHECK_LE(0, filterCol);
                CHECK_GT(filters.size(0), filter);
                CHECK_GT(filters.size(1), plane);
                CHECK_GT(filters.size(2), filterRow);
                CHECK_GT(filters.size(3), filterCol);
                const auto filt =
                  filters.at({filter, plane, filterRow, filterCol});

                const auto val = filt * in;
                output.at({batch, filter, outputRow, outputCol}) += val;
              }
            }
          }
        }
      }
    }
  }
  return output;
}


Tensor<float>
convolutionFull(
  const Tensor<float>& output,
  const Tensor<float>& filters,
  long filterRowStride,
  long filterColStride,
  const folly::Optional<std::tuple<long, long, long, long>>& padding) {

  const long inputPaddingTop = padding ? get<0>(*padding) : 0;
  const long inputPaddingBottom = padding ? get<1>(*padding) : 0;
  const long inputPaddingLeft = padding ? get<2>(*padding) : 0;
  const long inputPaddingRight = padding ? get<3>(*padding) : 0;

  if (output.ndims() != 4 || filters.ndims() != 4) {
    throw std::invalid_argument("illegal output/filter format");
  }

  if (output.size(1) != filters.size(0)) {
    throw std::invalid_argument("num planes mismatch");
  }

  if (filterRowStride < 1 || filterColStride < 1) {
    throw std::invalid_argument("filter stride should be >= 1");
  }

  const auto inputRowsAfterPadding =
    getFullConvSize(output.size(2), filters.size(2), filterRowStride);
  const auto inputRowsBeforePadding =
    inputRowsAfterPadding - inputPaddingTop - inputPaddingBottom;

  const auto inputColsAfterPadding =
    getFullConvSize(output.size(3), filters.size(3), filterColStride);
  const auto inputColsBeforePadding =
    inputColsAfterPadding - inputPaddingLeft - inputPaddingRight;

  auto input =
    Tensor<float>{{output.size(0), filters.size(1),
                   inputRowsBeforePadding, inputColsBeforePadding}};
  input.fill(0);

  // doall
  for (long batch = 0; batch < output.size(0); ++batch) {
    // doall
    for (long inputPlane = 0; inputPlane < filters.size(1); ++inputPlane) {
      // reduct
      for (long filterPlane = 0; filterPlane < filters.size(0); ++filterPlane) {

        // doall
        for (long inputRowBeforePadding = 0;
             inputRowBeforePadding < input.size(2);
             ++inputRowBeforePadding) {
          const auto inputRowAfterPadding =
            inputRowBeforePadding + inputPaddingTop;

          // reduct
          // input = output * stride + filter, so input - filter
          // must be a whole multiple of stride. outputRow must
          // be non-negative as well.
          for (long filterRow = inputRowAfterPadding % filterRowStride;
               filterRow <= inputRowAfterPadding && filterRow < filters.size(2);
               filterRow += filterRowStride) {
            const auto outputRow =
              (inputRowAfterPadding - filterRow) / filterRowStride;

            // The upper bound of the valid area convolution,
            // based on what filter points we are using
            if (outputRow >= output.size(2)) {
              continue;
            }

            // doall
            for (long inputColBeforePadding = 0;
                 inputColBeforePadding < input.size(3);
                 ++inputColBeforePadding) {
              const auto inputColAfterPadding =
                inputColBeforePadding + inputPaddingLeft;

              // reduct
              // input = output * stride + filter, so input - filter
              // must be a whole multiple of stride. outputCol must
              // be non-negative as well.
              for (long filterCol = inputColAfterPadding % filterColStride;
                   filterCol <= inputColAfterPadding &&
                     filterCol < filters.size(3);
                   filterCol += filterColStride) {
                const auto outputCol =
                  ((inputColAfterPadding - filterCol) / filterColStride);

                // The upper bound of the valid area convolution,
                // based on what filter points we are using
                if (outputCol >= output.size(3)) {
                  continue;
                }

                const auto filt =
                  filters.at({filterPlane, inputPlane, filterRow, filterCol});

                const auto out =
                  output.at({batch, filterPlane, outputRow, outputCol});

                input.at({batch, inputPlane,
                          inputRowBeforePadding,
                          inputColBeforePadding}) += filt * out;
              }
            }
          }
        }
      }
    }
  }

  return input;
}

Tensor<float>
crossCorrelationReverseValidOnly(
  const Tensor<float>& input,
  const Tensor<float>& output,
  long filterRowStride,
  long filterColStride,
  float scale,
  const folly::Optional<std::tuple<long, long, long, long>>& padding) {

  const long inputPaddingTop = padding ? get<0>(*padding) : 0;
  const long inputPaddingBottom = padding ? get<1>(*padding) : 0;
  const long inputPaddingLeft = padding ? get<2>(*padding) : 0;
  const long inputPaddingRight = padding ? get<3>(*padding) : 0;

  if (input.ndims() != 4 || output.ndims() != 4) {
    throw std::invalid_argument("illegal input/output format");
  }

  if (input.size(0) != output.size(0)) {
    throw std::invalid_argument("batch size mismatch");
  }

  if (filterRowStride < 1 || filterColStride < 1) {
    throw std::invalid_argument("filter stride should be >= 1");
  }

  // The filter must fit in an integral number of times into the image
  // after the stride
  if (output.size(2) > input.size(2) + inputPaddingTop + inputPaddingBottom) {
    throw std::invalid_argument("output rows must be <= input rows in size");
  }

  if (output.size(3) > input.size(3) + inputPaddingLeft + inputPaddingRight) {
    throw std::invalid_argument("output cols must be <= input cols in size");
  }

  const auto inputRowsAfterPadding =
    input.size(2) + inputPaddingTop + inputPaddingBottom;
  const auto inputColsAfterPadding =
    input.size(3) + inputPaddingLeft + inputPaddingRight;

  const auto filterRows =
    getValidRevConvSize(inputRowsAfterPadding, output.size(2), filterRowStride);
  const auto filterCols =
    getValidRevConvSize(inputColsAfterPadding, output.size(3), filterColStride);

  auto filters =
    Tensor<float>{{output.size(1), input.size(1), filterRows, filterCols}};
  filters.fill(0);

  // reduct
  for (long batch = 0; batch < input.size(0); ++batch) {
    // doall
    for (long outputPlane = 0; outputPlane < output.size(1); ++outputPlane) {
      // doall
      for (long inputPlane = 0; inputPlane < input.size(1); ++inputPlane) {
        // reduct
        for (long outputRow = 0; outputRow < output.size(2); ++outputRow) {
          // reduct
          for (long outputCol = 0; outputCol < output.size(3); ++outputCol) {
            const auto out =
              output.at({batch, outputPlane, outputRow, outputCol});

            // doall
            for (long filtersRow = 0; filtersRow < filterRows; ++filtersRow) {
              const auto inputRowAfterPadding =
                outputRow * filterRowStride + filtersRow;
              const auto inputRowBeforePadding =
                inputRowAfterPadding - inputPaddingTop;

              if (inputRowBeforePadding < 0 ||
                  inputRowBeforePadding >= input.size(2)) {
                continue;
              }

              // doall
              for (long filtersCol = 0; filtersCol < filterCols; ++filtersCol) {
                const auto inputColAfterPadding =
                  outputCol * filterColStride + filtersCol;
                const auto inputColBeforePadding =
                  inputColAfterPadding - inputPaddingLeft;

                if (inputColBeforePadding < 0 ||
                    inputColBeforePadding >= input.size(3)) {
                  continue;
                }

                const auto in =
                  input.at({batch, inputPlane,
                            inputRowBeforePadding, inputColBeforePadding});

                filters.at({outputPlane, inputPlane, filtersRow, filtersCol}) +=
                  scale * out * in;
              }
            }
          }
        }
      }
    }
  }

  return filters;
}

} } } } // namespace

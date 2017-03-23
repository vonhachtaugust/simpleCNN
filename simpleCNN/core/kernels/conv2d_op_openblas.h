//
// Created by hacht on 3/3/17.
//

#pragma once

#include "../../layers/convolutional_layer.h"
#include "../../util/im2col2im.h"
#include "../../util/util.h"
#include "../framework/tensor_utils.h"

namespace simpleCNN {
  namespace kernels {

    // out_data shape determined
    inline void conv2d_op_openblas(const tensor_t& in_data,
                                   const tensor_t& weights,
                                   const tensor_t& bias,
                                   tensor_t& out_data,
                                   const core::Conv_params& params) {
      matrix_t mRows({params.out_channels, params.in_channels * params.filter_height * params.filter_width});
      matrix_t mCols(
        {params.in_channels * params.filter_height * params.filter_width, params.output_height * params.output_width});
      matrix_t mResult({mRows.rows(), mCols.cols()});

      // Each weight is turned into a row.
      im2row_cpu(weights, mRows, params.out_channels, weights.dimension(dim_t::depth), weights.dimension(dim_t::height),
                 weights.dimension(dim_t::width));

      for (size_t i = 0; i < params.batch_size; ++i) {
        // Each position on which the weight will be applied is turned into a column.
        im2col_cpu(in_data, i, mCols, params.in_channels, params.input_height, params.input_width, params.filter_size(),
                   params.stride(), params.padding);
        multiply_2_dim_tensors_float(mRows, mCols, mResult, false, false);
        col2im_cpu(mResult, i, out_data, params.out_channels, params.output_height, params.output_width);
      }
    }
  }  // namespace kernels
}  // namespace simpleCNN

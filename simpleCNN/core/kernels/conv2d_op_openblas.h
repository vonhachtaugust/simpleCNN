//
// Created by hacht on 3/3/17.
//

#pragma once

#include "../../layers/convolutional_layer.h"
#include "../../util/im2col2im.h"
#include "../../util/util.h"

namespace simpleCNN {
  namespace kernels {

    inline void conv2d_op_openblas(const tensor_t& in_data,
                                   const tensor_t& weights,
                                   const tensor_t& bias,
                                   tensor_t& out_data,
                                   const core::Conv_params& params) {
      // Setup matrice sizes
      size_t vertical_locations =
        params.conv_out_length(in_data.height(), params.weights.height(),
                               params.vertical_stride, params.padding);
      size_t horizontal_locations =
        params.conv_out_length(in_data.width(), params.weights.width(),
                               params.horizontal_stride, params.padding);

      matrix_t m_col(
        {weights.size(), vertical_locations * horizontal_locations});
      matrix_t m_row({weights.depth(), weights.size()});
      matrix_t m_result(
        {weights.depth(), vertical_locations * horizontal_locations});

      // Convert in_data and weights into matrices
      im2col_cpu(in_data, m_col, params.channels(), in_data.height(),
                 in_data.width(), params.filter_size(), params.stride(),
                 params.padding);
      im2col_cpu(weights, m_row, params.channels(), params.weights.height(),
                 params.weights.width());

      // perform matrix multiplication
      multiply_2_dim_tensors_float(m_row, m_col, m_result, false, false);

      // Convert result into tensor and assign to out_data
      col2im_cpu(m_result, out_data, out_data.depth(), out_data.height(),
                 out_data.width());
    }
  }  // namespace kernels
}  // namespace simpleCNN

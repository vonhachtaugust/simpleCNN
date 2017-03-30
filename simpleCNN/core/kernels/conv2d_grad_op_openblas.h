//
// Created by hacht on 3/27/17.
//

#pragma once

#include "../../util/im2col2im.h"

namespace simpleCNN {
  namespace kernels {

    /**
     * @brief Performs element-wise matrix multiplication between the weights
     * (transposed) and previous layer deltas. The result is this layer deltas.
     *
     * @param mWeights weights in row format
     * @param mPrev_delta previous layer deltas as column format
     * @param mResult intermediate result in matrix format
     * @param batch_number
     * @param params
     * @param data_gradient mResult in tensor format
     */
    inline void get_gradient_for_data(matrix_t& mWeights,
                                      matrix_t& mPrev_delta,
                                      matrix_t& mResult,
                                      tensor_t& data_gradient,
                                      const int batch_number,
                                      const core::Conv_params& params) {
      multiply_2_dim_tensors_float(mWeights, mPrev_delta, mResult, true, false);
      col2im_cpu(mResult, batch_number, data_gradient, params.in_channels, params.input_height, params.input_width);
    }

    /**
     * @breif Performs element-wise matrix multiplication between the
     * previous layer deltas and the incoming data (transposed). The
     * result is this layer dW.
     *
     * @param input_from_previous_layer input (in forward pass sense)
     * @param mPrev_delta input (in backward pass sense)
     * @param dW weight gradients
     * @param batch_number
     * @param params
     */
    inline void get_gradient_for_filters(const tensor_t& input_from_previous_layer,
                                         matrix_t& mPrev_delta,
                                         tensor_t& dW,
                                         const int batch_number,
                                         const core::Conv_params& params) {
      matrix_t mCols(
        {params.in_channels * params.filter_height * params.filter_width, params.output_height * params.output_width});
      matrix_t mFilterResult({params.out_channels, params.in_channels * params.filter_height * params.filter_width});
      im2col_cpu(input_from_previous_layer, batch_number, mFilterResult, params.in_channels, params.input_height,
                 params.input_width);

      multiply_2_dim_tensors_float(mPrev_delta, mCols, mFilterResult, false, true);
      col2im_cpu(mFilterResult, batch_number, dW, params.in_channels, params.filter_height, params.filter_width);
    }

    /**
     * @brief Computes the data, weight and bias gradients used for backpropagation of signal error.
     *
     * @details data = prev_delta * flip(weights)
     * @details gradW = input_from_previous_layer * prev_delta
     * @details bias = prev_delta
     * @note The * symbol above denotes convolution.
     *
     * @param input_from_previous_layer
     * @param prev_delta
     * @param weights
     * @param curr_delta
     * @param params
     */
    inline void conv2d_grad_op_openblas(const tensor_t& input_from_previous_layer,
                                        const tensor_t& weights,
                                        const tensor_t& prev_delta,
                                        tensor_t& dW,
                                        tensor_t& dB,
                                        tensor_t& curr_delta,
                                        const core::Conv_params& params) {
      // Some calculations which resides inside 'get' functions are performed here in order
      // to minimize overhead.
      matrix_t mCols({params.out_channels, params.output_height * params.output_width});  // prev_delta matrix
      matrix_t mRows(
        {params.out_channels, params.in_channels * params.filter_height * params.filter_width});  // weight matrix
      matrix_t mDataResult({mRows.cols(), mCols.cols()});                                         // intermediate result

      // Each weight is flipped and turned into a list of row
      im2row_flipped_cpu(weights, mRows, params.out_channels, weights.dimension(dim_t::depth),
                         weights.dimension(dim_t::height), weights.dimension(dim_t::width));

      for (size_t i = 0; i < params.batch_size; ++i) {
        // Each previous delta is turned into a list of columns
        im2col_cpu(prev_delta, i, mCols, params.out_channels, params.output_height, params.output_width);

        get_gradient_for_data(mRows, mCols, mDataResult, curr_delta, i, params);
        get_gradient_for_filters(input_from_previous_layer, mCols, dW, i, params);
        // get_gradient_for_bias
      }
    }
  }  // namespace kernels
}  // namespace simpleCNN

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

    inline void conv2d_op_openblas(const tensor_t& in_data,
                                   const tensor_t& weights,
                                   const tensor_t& bias,
                                   tensor_t& out_data,
                                   const core::Conv_params& params) {
      matrix_t mRows(
        {params.out_channels,
         params.in_channels * params.filter_height * params.filter_width});
      matrix_t mCols(
        {params.in_channels * params.filter_height * params.filter_width,
         params.output_height * params.output_width});
      matrix_t mResult({mRows.rows(), mCols.cols()});

      // Each weight is turned into a row.
      im2row_cpu(
        weights, mRows, params.out_channels, weights.dimension(dim_t::depth),
        weights.dimension(dim_t::height), weights.dimension(dim_t::width));

      for (size_t i = 0; i < params.batch_size; ++i) {
        // Each position on which the weight will be applied is turned into a
        // column.
        im2col_cpu(in_data, i, mCols, params.in_channels, params.input_height,
                   params.input_width, params.filter_size(), params.stride(),
                   params.padding);
        multiply_2_dim_tensors_float(mRows, mCols, mResult, false, false);
        col2im_cpu(mResult, i, out_data, params.out_channels,
                   params.output_height, params.output_width);

        if (params.has_bias) {
          for (size_t j = 0; j < out_data.dimension(dim_t::depth); ++j) {
            auto start = out_data.host_iter(i, j, 0, 0);
            auto end   = start +
                       out_data.dimension(dim_t::height) *
                         out_data.dimension(dim_t::width);
            auto val = bias.host_at(j, i, 0, 0);
            for (; start != end; ++start) {
              *start += val;
            }
          }
        }
      }
    }

    /**
       * @brief Computes the data, weight and bias gradients used for
     * backpropagation of signal error.
       *
       * @details data = flip(weights) * prev_delta
       * @details gradW = previous_layer_input * prev_delta
       * @details bias = prev_delta
       * @note The * symbol above denotes convolution.
       *
       * @param previous_layer_input
       * @param weights
       * @param dW
       * @param dB
       * @param prev_delta
       * @param curr_delta
       * @param params
       */
    inline void conv2d_op_openblas(const tensor_t& previous_layer_input,
                                   const tensor_t& weights,
                                   tensor_t& dW,
                                   tensor_t& dB,
                                   tensor_t& prev_delta,
                                   tensor_t& curr_delta,
                                   const core::Conv_params& params) {
      // Matrices for backpropagation of deltas;
      matrix_t mWeights(
        {params.in_channels,
         params.filter_height * params.filter_width * params.out_channels});
      matrix_t mCurr_delta(
        {params.out_channels * params.output_width * params.output_height,
         params.input_width * params.input_height});
      matrix_t mResult_delta({mWeights.rows(), mCurr_delta.cols()});

      // Matrices for backpropagation of weights;
      matrix_t mPrev_layer_input(
        {params.in_channels * params.filter_height * params.filter_width,
         params.output_width * params.output_height});
      matrix_t mCurr_delta_no_pad(
        {params.out_channels, params.output_height * params.output_width});
      matrix_t mResult_dW(
        {mPrev_layer_input.rows(), mCurr_delta_no_pad.rows()});

      im2row_flipped_cpu(weights, mWeights, params.out_channels,
                         params.in_channels, params.filter_height,
                         params.filter_width);
      for (size_t i = 0; i < params.batch_size; ++i) {
        // Backpropagation delta - current layer delta to previous layer delta.
        im2col_cpu(curr_delta, i, mCurr_delta, params.out_channels,
                   params.output_height, params.output_width,
                   params.filter_size(), 1, params.filter_size() - 1);
        multiply_2_dim_tensors_float(mWeights, mCurr_delta, mResult_delta,
                                     false, false);
        col2im_cpu(mResult_delta, i, prev_delta, params.in_channels,
                   params.input_height, params.input_width);

        // Backpropagation dW - prev layer input conv. curr deltas transposed
        im2col_cpu(previous_layer_input, i, mPrev_layer_input,
                   params.in_channels, params.input_height, params.input_width,
                   params.filter_size(), params.stride(), params.padding);
        im2col_cpu(curr_delta, i, mCurr_delta_no_pad, params.out_channels,
                   params.output_height, params.output_width);

        multiply_2_dim_tensors_float(mPrev_layer_input, mCurr_delta_no_pad,
                                     mResult_dW, false, true);
        row2im_cpu(mResult_dW, dW, params.out_channels, params.in_channels,
                   params.filter_height, params.filter_width);

        // Backpropagation dB - curr delta = dB
        for (size_t j = 0; j < curr_delta.dimension(dim_t::depth); ++j) {
          auto start = curr_delta.host_iter(i, j, 0, 0);
          auto end   = start +
                     curr_delta.dimension(dim_t::width) *
                       curr_delta.dimension(dim_t::height);
          dB.host_at(j, i, 0, 0) = std::accumulate(start, end, float_t(0));
        }
      }
    }
  }  // namespace kernels
}  // namespace simpleCNN

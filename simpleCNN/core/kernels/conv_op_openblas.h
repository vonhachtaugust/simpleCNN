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

    /**
     * Forward pass --
     *
     * @param in_data
     * @param weights
     * @param bias
     * @param out_data
     * @param params
     */
    inline void conv_op_openblas(const tensor_t& in_data,
                                 const tensor_t& weights,
                                 const tensor_t& bias,
                                 tensor_t& out_data,
                                 const core::Conv_params& params) {
      matrix_t mRows({params.out_channels, params.in_channels * params.filter_height * params.filter_width});
      matrix_t mCols(
        {params.in_channels * params.filter_height * params.filter_width, params.output_height * params.output_width});
      matrix_t mResult({mRows.shape()[0], mCols.shape()[1]});

      im2row_cpu(weights, mRows, params.out_channels, weights.dimension(dim_t::depth), weights.dimension(dim_t::height),
                 weights.dimension(dim_t::width));
      for (size_t i = 0; i < params.batch_size; ++i) {
        im2col_cpu(in_data, i, mCols, params.in_channels, params.input_height, params.input_width, params.filter_size(),
                   params.stride(), params.padding);
        sgemm(mRows, mCols, mResult, false, false);
        col2im_insert_cpu(mResult, i, out_data, params.out_channels, params.output_height, params.output_width);

        if (params.has_bias) {
          for (size_t j = 0; j < out_data.dimension(dim_t::depth); ++j) {
            auto start = out_data.host_iter(i, j, 0, 0);
            auto end   = start + out_data.dimension(dim_t::height) * out_data.dimension(dim_t::width);
            auto val   = bias.host_at(j, 0, 0, 0);
            for (; start != end; ++start) {
              *start += val;
            }
          }
        }
      }
    }

    /**
     * Backpropagation
     *
     * @param weights
     * @param prev_delta
     * @param curr_delta
     * @param params
     */
    inline void backpropagate_deltas(const tensor_t& weights,
                                     tensor_t& prev_delta,
                                     tensor_t& curr_delta,
                                     const core::Conv_params& params) {
      matrix_t mWeights({params.in_channels, params.filter_height * params.filter_width * params.out_channels});
      matrix_t mCurr_delta(
        {params.out_channels * params.filter_height * params.filter_width, params.input_width * params.input_height});
      matrix_t mResult_delta({mWeights.shape()[0], mCurr_delta.shape()[1]});

      im2row_flipped_cpu(weights, mWeights, params.out_channels, params.in_channels, params.filter_height,
                         params.filter_width);
      for (size_t i = 0; i < params.batch_size; ++i) {
        im2col_cpu(curr_delta, i, mCurr_delta, params.out_channels, params.output_height, params.output_width,
                   params.filter_size(), 1, params.filter_size() - 1);
        sgemm(mWeights, mCurr_delta, mResult_delta, false, false);
        col2im_insert_cpu(mResult_delta, i, prev_delta, params.in_channels, params.input_height, params.input_width);
      }
    }

    /**
     * Computes dW and db.
     *
     * @param prev_in
     * @param weight
     * @param dW
     * @param db
     * @param curr_delta
     * @param params
     */
    inline void accumulate_deltas(const tensor_t& prev_in,
                                  const tensor_t& weight,
                                  tensor_t& dW,
                                  tensor_t& db,
                                  const tensor_t& curr_delta,
                                  const core::Conv_params& params) {
      matrix_t mPrev_in(
        {params.in_channels * params.filter_height * params.filter_width, params.output_width * params.output_height});
      matrix_t mCurr_delta({params.out_channels, params.output_height * params.output_width});
      matrix_t mResult_dW({mPrev_in.shape()[0], mCurr_delta.shape()[0]});

      for (size_t i = 0; i < params.batch_size; ++i) {
        im2col_cpu(prev_in, i, mPrev_in, params.in_channels, params.input_height, params.input_width,
                   params.filter_size(), params.stride(), params.padding);
        im2col_cpu(curr_delta, i, mCurr_delta, params.out_channels, params.output_height, params.output_width);

        sgemm(mPrev_in, mCurr_delta, mResult_dW, false, true);

        // Add up dW instead of merge later. Average value is used later and the division is performed at that point.
        row2im_add_cpu(mResult_dW, dW, params.out_channels, params.in_channels, params.filter_height,
                       params.filter_width);
        if (params.has_bias) {
          for (size_t j = 0; j < params.out_channels; ++j) {
            auto start = curr_delta.host_iter(i, j, 0, 0);
            auto end   = start + params.output_height * params.output_width;

            db.host_at(j, 0, 0, 0) += std::accumulate(start, end, float_t(0));
          }
        }
      }
      average_deltas(dW, params.batch_size);
      average_deltas(db, params.batch_size);
    }

    /**
     * Backward pass --
     *
     * @param prev_in
     * @param weights
     * @param dW
     * @param db
     * @param prev_delta
     * @param curr_delta
     * @param params
     */
    inline void conv_op_openblas(const tensor_t& prev_in,
                                 const tensor_t& weights,
                                 tensor_t& dW,
                                 tensor_t& db,
                                 tensor_t& prev_delta,
                                 tensor_t& curr_delta,
                                 const core::Conv_params& params) {
      backpropagate_deltas(weights, prev_delta, curr_delta, params);
      accumulate_deltas(prev_in, weights, dW, db, curr_delta, params);
    }
  }  // namespace kernels
}  // namespace simpleCNN

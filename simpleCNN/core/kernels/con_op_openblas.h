//
// Created by hacht on 4/5/17.
//

#pragma once

#include "../params/con_params.h"

namespace simpleCNN {
  namespace kernels {

    /**
     * Forward pass
     *
     * @param in_data
     * @param weight
     * @param bias
     * @param out_data
     * @param params
     */
    inline void con_op_openblas(const tensor_t &in_data,
                                const tensor_t &weight,
                                const tensor_t &bias,
                                tensor_t &out_data,
                                const core::Con_params &params) {
      matrix_t mRows({params.out_dim, params.in_dim});
      matrix_t mCols({params.in_dim, 1});
      matrix_t mResult({params.out_dim, 1});

      im2mat_cpu(weight, mRows, 0, params.out_dim, params.in_dim);
      for (size_t i = 0; i < params.batch_size; ++i)
      {
        im2mat_cpu(in_data, mCols, i, params.in_dim, 1);
        sgemm(mRows, mCols, mResult, false, false);
        mat2im_cpu(mResult, out_data, i, params.out_dim, 1);

        if (params.has_bias) {
          for (size_t j = 0; j < params.out_dim; ++j) {
            out_data.host_at(i, 0, j, 0) += bias.host_at(i, 0, j, 0);
          }
        }
      }
    }

    /**
     * Backpropagates delta
     *
     * @param weight
     * @param curr_delta
     * @param prev_delta
     * @param params
     */
    inline void con_op_openblas(const tensor_t &weight,
                                tensor_t &curr_delta,
                                tensor_t &prev_delta,
                                const core::Con_params &params) {
      matrix_t mRows({params.out_dim, params.in_dim});
      matrix_t mCols({params.out_dim, 1});
      matrix_t mResult({params.in_dim, 1});

      im2mat_cpu(weight, mRows, 0, params.out_dim, params.in_dim);
      for (size_t i = 0; i < params.batch_size; ++i) {
        im2mat_cpu(curr_delta, mCols, i, params.out_dim, 1);
        sgemm(mRows, mCols, mResult, true, false);
        mat2im_cpu(mResult, prev_delta, i, params.in_dim, 1);
      }
    }

    /**
     * Computes dW and db
     *
     * @param prev_in
     * @param weight
     * @param dW
     * @param db
     * @param curr_delta
     * @param params
     */
    inline void con_op_openblas(const tensor_t& prev_in,
                                const tensor_t& weight,
                                tensor_t& dW,
                                tensor_t& db,
                                const tensor_t& curr_delta,
                                const core::Con_params& params) {
      matrix_t mRows({params.in_dim, 1});
      matrix_t mCols({params.out_dim, 1});
      matrix_t mResult({params.in_dim, params.out_dim});

      for (size_t i = 0; i < params.batch_size; ++i)
      {
        im2mat_cpu(prev_in ,mRows, i, params.in_dim, 1);
        im2mat_cpu(curr_delta, mCols, i, params.out_dim, 1);
        sgemm(mRows, mCols, mResult, false, true);
        mat2im_cpu(mResult, dW, i, params.in_dim, params.out_dim);

        if (params.has_bias) {
          for (size_t j = 0; j < params.out_dim; ++j) {
            db.host_at(i, 0, j, 0) = curr_delta.host_at(i, 0, j, 0);
          }
        }
      }
    }

  } // namespace kernels
} // namespace simpleCNN
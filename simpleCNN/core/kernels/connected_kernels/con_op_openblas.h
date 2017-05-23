//
// Created by hacht on 4/5/17.
//

#pragma once

#include "../../params/con_params.h"

namespace simpleCNN {
  namespace kernels {

    /**
     * Forward pass --
     *
     * @param in_data
     * @param weight
     * @param bias
     * @param out_data
     * @param params
     */
    inline void con_op_openblas(const tensor_t& in_data,
                                const tensor_t& weight,
                                const tensor_t& bias,
                                tensor_t& out_data,
                                const core::Con_params& params) {
      for (size_t i = 0; i < params.batch_size; ++i) {
        auto start_weight = weight.host_begin();
        auto start_in     = in_data.host_ptr(i, 0, 0, 0);
        auto start_out    = out_data.host_ptr(i, 0, 0, 0);

        multiply(weight, &(*start_weight), in_data, start_in, start_out, false, false);
        if (params.has_bias) {
          for (size_t j = 0; j < params.out_dim; ++j) {
            out_data.host_at(i, 0, j, 0) += bias.host_at(0, 0, j, 0);
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
    inline void backpropagate_deltas(const tensor_t& weight,
                                     tensor_t& curr_delta,
                                     tensor_t& prev_delta,
                                     const core::Con_params& params) {
      for (size_t i = 0; i < params.batch_size; ++i) {
        auto start_weight = weight.host_begin();
        auto start_curr   = curr_delta.host_ptr(i, 0, 0, 0);
        auto start_prev   = prev_delta.host_ptr(i, 0, 0, 0);

        multiply(weight, &(*start_weight), curr_delta, start_curr, start_prev, true, false);
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
    inline void accumulate_deltas(const tensor_t& prev_in,
                                  const tensor_t& weight,
                                  tensor_t& dW,
                                  tensor_t& db,
                                  tensor_t& curr_delta,
                                  const core::Con_params& params) {
      tensor_t result({params.batch_size, dW.shape()[1], dW.shape()[2], dW.shape()[3]});
      for (size_t i = 0; i < params.batch_size; ++i) {
        auto start_prev   = prev_in.host_ptr(i, 0, 0, 0);
        auto start_curr   = curr_delta.host_ptr(i, 0, 0, 0);
        auto start_result = result.host_ptr(i, 0, 0, 0);

        multiply(curr_delta, start_curr, prev_in, start_prev, start_result, false, true);
        if (params.has_bias) {
          for (size_t j = 0; j < params.out_dim; ++j) {
            size_t index = i * params.out_dim + j;
            db.host_at_index(j) += curr_delta.host_at_index(index);
          }
        }
      }

      size_t length = result.size() / params.batch_size;
      for (size_t i = 0; i < params.batch_size; ++i) {
        for (size_t j = 0; j < length; ++j) {
          size_t index = i * length + j;
          dW.host_at_index(j) += result.host_at_index(index);  // / static_cast<float_t>(params.batch_size);
        }
      }
      // average_deltas(db, params.batch_size);
      // average_deltas(dW, params.batch_size);
    }

    /**
     * Backward pass --
     *
     * @param prev_in
     * @param weight
     * @param dW
     * @param db
     * @param curr_delta
     * @param prev_delta
     * @param params
     */
    inline void con_op_openblas(const tensor_t& prev_in,
                                const tensor_t& weight,
                                tensor_t& dW,
                                tensor_t& db,
                                tensor_t& curr_delta,
                                tensor_t& prev_delta,
                                const core::Con_params& params) {
      backpropagate_deltas(weight, curr_delta, prev_delta, params);
      accumulate_deltas(prev_in, weight, dW, db, curr_delta, params);
    }
  }  // namespace kernels
}  // namespace simpleCNN
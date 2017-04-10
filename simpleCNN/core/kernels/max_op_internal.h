//
// Created by hacht on 4/3/17.
//

#pragma once

#include "../../layers/maxpooling_layer.h"
//#include "../../util/util.h"
//#include "../framework/tensor_utils.h"

namespace simpleCNN {
  namespace kernels {

    /**
     * Forward pass
     *
     * @param in_data
     * @param out_data
     * @param params
     */
    inline void maxpooling_op_internal(const tensor_t& in_data,
                                       tensor_t& out_data,
                                       const core::Maxpooling_params& params) {
      for (size_t b = 0; b < params.batch_size; ++b) {
        for (size_t ch = 0; ch < params.out_channels; ++ch) {
          for (size_t i = 0; i < params.output_height; ++i) {
            for (size_t j = 0; j < params.output_width; ++j) {
              float max    = std::numeric_limits<float_t>::lowest();
              size_t max_i = -1;
              for (size_t n = 0; n < params.pooling_size_y; ++n) {
                for (size_t m = 0; m < params.pooling_size_x; ++m) {
                  size_t y = i * params.stride_y + n;
                  size_t x = j * params.stride_x + m;

                  float val = in_data.host_at(b, ch, y, x);
                  max_i = (val > max) ? (m+ n * params.pooling_size_x) : max_i;
                  max   = (val > max) ? val : max;
                }
              }
              *params.max_index.host_iter(b, ch, i, j) = max_i;
              *out_data.host_iter(b, ch, i, j)         = max;
            }
          }
        }
      }
    }

    /**
     * Backpropagation
     *
     * @param curr_delta
     * @param prev_delta
     * @param max_index
     * @param params
     */
    inline void maxpooling_op_internal(const tensor_t& curr_delta,
                                       tensor_t& prev_delta,
                                       tensor_t& max_index,
                                       const core::Maxpooling_params& params) {
      for (size_t b = 0; b < params.batch_size; ++b) {
        for (size_t ch = 0; ch < params.out_channels; ++ch) {
          for (size_t i = 0; i < params.output_height; ++i) {
            for (size_t j = 0; j < params.output_width; ++j) {
              size_t max_i = max_index.host_at(b, ch, i, j);
              for (size_t n = 0; n < params.pooling_size_y; ++n) {
                for (size_t m = 0; m < params.pooling_size_x; ++m) {
                  size_t y     = i * params.stride_y + n;
                  size_t x     = j * params.stride_x + m;
                  size_t index = m + params.stride_x * n;

                  *prev_delta.host_iter(b, ch, y, x) =
                    (max_i == index) ? *curr_delta.host_iter(b, ch, i, j)
                                     : float_t{0};
                }
              }
            }
          }
        }
      }
    }
  }  // namespace kernels
}  // namespace simpleCNN
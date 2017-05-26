//
// Created by hacht on 5/22/17.
//

#pragma once

#include "../../../util/random.h"
#include "../../../util/util.h"
#include "../../params/dropout_params.h"

namespace simpleCNN {
  namespace kernels {

    inline void dropout_op_internal(const tensor_t& in_data, tensor_t& out_data, const core::Dropout_params& params) {
      const size_t n = in_data.size();
      if (params.phase == net_phase::train) {
        for (size_t i = 0; i < n; ++i) {
          params.mask.host_at_index(i) = bernoulli(params.prob);
          out_data.host_at_index(i)    = in_data.host_at_index(i) * params.mask.host_at_index(i);
        }
      } else {
        for (size_t i = 0; i < n; ++i) {
          // Approximate the output by expected input value.
          out_data.host_at_index(i) = in_data.host_at_index(i) * (float_t(1) - params.prob);
        }
      }
    }

    inline void dropout_op_internal(const tensor_t& curr_delta,
                                    tensor_t& prev_delta,
                                    const core::Dropout_params& params,
                                    const size_t n) {
      for (size_t i = 0; i < n; ++i) {
        prev_delta.host_at_index(i) = params.mask.host_at_index(i) * curr_delta.host_at_index(i);
      }
    }
  }  // namespace kernels
}  // namespace simpleCNN

//
// Created by hacht on 5/22/17.
//

#pragma once

#include "../../../util/util.h"
#include "../../framework/op_kernel.h"
#include "../../params/activation_params.h"

namespace simpleCNN {
  namespace kernels {

    inline void relu_forward(const tensor_t& in_data,
                             tensor_t& out_data) {
      for (size_t i = 0; i < in_data.size(); ++i) {
          float_t val = *(in_data.host_begin() + i);
          out_data.host_at_index(i) = std::max(float_t(0), val);
      }
    }

    inline void relu_backward(const tensor_t& in_data,
                              const tensor_t& curr_delta,
                              tensor_t& prev_delta) {
      for (size_t i = 0; i < in_data.size(); ++i) {
        auto value = in_data.host_at_index(i) > float_t(0) ? float_t(1) : float_t(0);
        prev_delta.host_at_index(i) = value * curr_delta.host_at_index(i);
      }
    }

    inline void activation_op_internal(const tensor_t& in_data,
                                       const tensor_t& curr_delta,
                                       tensor_t& prev_delta,
                                       const core::activation_t h) {
      if (h == core::activation_t::relu) {
        relu_backward(in_data, curr_delta, prev_delta);
      } else {
        throw simple_error("Not a supported activation function");
      }
    }

    inline void activation_op_internal(const tensor_t& in_data,
                                       tensor_t& out_data,
                                       const core::activation_t h) {
      if (h == core::activation_t::relu) {
        relu_forward(in_data, out_data);
      } else {
        throw simple_error("Not a supported activation function");
      }
    }
  } // namespace kernels
} // namespace simpleCNN
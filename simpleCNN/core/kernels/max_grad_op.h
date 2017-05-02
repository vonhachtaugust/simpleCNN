//
// Created by hacht on 4/4/17.
//

#pragma once

#include "../params/max_params.h"
#include "max_op_internal.h"

namespace simpleCNN {
  class MaxpoolingGradOp : public core::OpKernel {
   public:
    explicit MaxpoolingGradOp(const core::OpKernelConstruction& context) : core::OpKernel(context) {}

    void compute(const core::OpKernelContext& context) override {
      core::Maxpooling_params* maxpooling_params_ptr = static_cast<core::Maxpooling_params*>(core::OpKernel::params_);
      const auto& params                             = maxpooling_params_ptr->maxpool();

      const tensor_t& curr_delta = context.output_grad(1);
      tensor_t& prev_delta       = context.input_grad(0);
      tensor_t& max_index        = const_cast<tensor_t&>(params.max_index);

      const core::backend_t engine = context.engine();

      if (engine == core::backend_t::internal) {
        kernels::maxpooling_op_internal(curr_delta, prev_delta, max_index, params);
      } else {
        throw simple_error("No supported engine: ");
      }
    }
  };
}
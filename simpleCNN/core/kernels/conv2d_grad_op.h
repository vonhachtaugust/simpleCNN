//
// Created by hacht on 3/27/17.
//

#pragma once

#include "conv2d_grad_op_openblas.h"

namespace simpleCNN {

  class Conv2dGradOp : public core::OpKernel {
   public:
    explicit Conv2dGradOp(const core::OpKernelConstruction &context) : core::OpKernel(context) {}

    void compute(const core::OpKernelContext &context) override {
      core::Conv_params *conv_params_ptr = static_cast<core::Conv_params *>(core::OpKernel::params_);
      const auto &params                 = conv_params_ptr->conv();

      const tensor_t &input_from_previous_layer = context.input(0);  // for weight gradient
      const tensor_t &weight   = context.input(1);  // for delta gradient
      tensor_t &prev_delta     = context.input_grad(0);
      tensor_t &dW             = context.input_grad(1);
      tensor_t &dB             = context.input_grad(2);
      tensor_t &curr_delta     = context.output_grad(0);

      const core::backend_t engine = context.engine();

      if (engine == core::backend_t::internal) {
        // fill dW and dB
      } else {
        throw simple_error("No supported engine: ");
      }
    }
  };
} // namespace simpleCNN

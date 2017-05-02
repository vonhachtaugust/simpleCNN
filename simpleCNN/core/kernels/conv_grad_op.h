//
// Created by hacht on 3/27/17.
//

#pragma once

#include "../params/conv_params.h"
#include "conv_op_openblas.h"

namespace simpleCNN {

  class ConvGradOp : public core::OpKernel {
   public:
    explicit ConvGradOp(const core::OpKernelConstruction &context) : core::OpKernel(context) {}

    void compute(const core::OpKernelContext &context) override {
      core::Conv_params *conv_params_ptr = static_cast<core::Conv_params *>(core::OpKernel::params_);
      const auto &params                 = conv_params_ptr->conv();

      const tensor_t &prev_in = context.input(0);
      const tensor_t &weight  = context.input(1);
      tensor_t &dW            = context.input_grad(1);
      tensor_t &db            = context.input_grad(2);
      tensor_t &prev_delta    = context.input_grad(0);
      tensor_t &curr_delta    = context.output_grad(1);
      //tensor_t &curr_delta_I  = context.output_grad(0);

      const core::backend_t engine = context.engine();
      if (engine == core::backend_t::internal) {
        kernels::conv_op_openblas(prev_in, weight, dW, db, prev_delta, curr_delta, params);
      } else {
        throw simple_error("No supported engine: ");
      }
    }
  };
}  // namespace simpleCNN

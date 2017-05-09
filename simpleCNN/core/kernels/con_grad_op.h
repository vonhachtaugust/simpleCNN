//
// Created by hacht on 4/5/17.
//

#pragma once

#include "../params/con_params.h"
#include "con_op_openblas.h"

namespace simpleCNN {
  class ConGradOp : public core::OpKernel {
   public:
    explicit ConGradOp(const core::OpKernelConstruction& context) : core::OpKernel(context) {}

    void compute(const core::OpKernelContext& context) override {
      core::Con_params* connected_params_ptr = static_cast<core::Con_params*>(core::OpKernel::params_);
      const auto& params                     = connected_params_ptr->connected_params();

      const tensor_t& prev_in = context.input(0);
      const tensor_t& weight  = context.input(1);
      tensor_t& dW            = context.input_grad(1);
      tensor_t& db            = context.input_grad(2);
      tensor_t& prev_delta    = context.input_grad(0);
      tensor_t& curr_delta    = context.output_grad(0);

      const core::backend_t engine = context.engine();

      if (engine == core::backend_t::internal) {
        kernels::con_op_openblas(prev_in, weight, dW, db, curr_delta, prev_delta, params);
      } else {
        throw simple_error("No supported engine");
      }
    }
  };
}
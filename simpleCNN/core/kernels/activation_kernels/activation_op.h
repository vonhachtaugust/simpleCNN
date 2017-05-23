//
// Created by hacht on 5/22/17.
//

#pragma once

#include "../../framework/op_kernel.h"
#include "activation_op_internal.h"
#include "../../params/activation_params.h"

namespace simpleCNN {

  class ActivationOp : public core::OpKernel {
   public:
    explicit ActivationOp(const core::OpKernelConstruction& context) : core::OpKernel(context) {}

    void compute(const core::OpKernelContext& context) override {
      core::Activation_params* active_params_ptr = static_cast<core::Activation_params*>(core::OpKernel::params_);
      const auto& params = active_params_ptr->activation_params();

      const tensor_t& in_data = context.input(0);
      tensor_t& out_data      = context.output(0);

      const core::backend_t engine = context.engine();
      const core::activation_t h = params.activation_function;

      if (engine == core::backend_t::internal) {
        kernels::activation_op_internal(in_data, out_data, h);
      } else {
       throw simple_error("No supported engine");
      }
    }
  };


class ActivationGradOp : public core::OpKernel {
  public:
  explicit ActivationGradOp(const core::OpKernelConstruction &context) : core::OpKernel(context) {}

  void compute(const core::OpKernelContext &context) override {
    core::Activation_params* active_params_ptr = static_cast<core::Activation_params*>(core::OpKernel::params_);
    const auto& params = active_params_ptr->activation_params();

    const tensor_t &input = context.input(0);
    const tensor_t &curr_delta    = context.output_grad(0);
    tensor_t& prev_delta  = context.input_grad(0);

    const core::backend_t engine = context.engine();
    const core::activation_t h = params.activation_function;

    if (engine == core::backend_t::internal) {
      kernels::activation_op_internal(input, curr_delta, prev_delta, h);
    } else {
      throw simple_error("No supported engine: ");
    }
  }
};
} // namespace simpleCNN

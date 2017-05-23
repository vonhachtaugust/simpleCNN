//
// Created by hacht on 5/22/17.
//

#pragma once

#include "../../framework/op_kernel.h"
#include "dropout_op_internal.h"

namespace simpleCNN {

  class DropoutOp : public core::OpKernel {
   public:
    explicit DropoutOp(const core::OpKernelConstruction& context) : core::OpKernel(context) {}

    void compute(const core::OpKernelContext& context) override {
      core::Dropout_params* drop_params_ptr = static_cast<core::Dropout_params*>(core::OpKernel::params_);
      const auto& params                    = drop_params_ptr->dropout_params();

      const tensor_t& in_data = context.input(0);
      tensor_t& out_data      = context.output(0);

      const core::backend_t engine = context.engine();
      if (engine == core::backend_t::internal) {
        kernels::dropout_op_internal(in_data, out_data, params);
      } else {
        throw simple_error("No supported engine");
      }
    }
  };

  class DropoutGradOp : public core::OpKernel {
   public:
    explicit DropoutGradOp(const core::OpKernelConstruction& context) : core::OpKernel(context) {}

    void compute(const core::OpKernelContext& context) override {
      core::Dropout_params* drop_params_ptr = static_cast<core::Dropout_params*>(core::OpKernel::params_);
      const auto& params                    = drop_params_ptr->dropout_params();

      const tensor_t& curr_delta = context.output_grad(0);
      tensor_t& prev_delta       = context.input_grad(0);

      const core::backend_t engine = context.engine();
      if (engine == core::backend_t::internal) {
        kernels::dropout_op_internal(curr_delta, prev_delta, params, curr_delta.size());
      } else {
        throw simple_error("No supported engine: ");
      }
    }
  };
}  // namespace simpleCNN

//
// Created by hacht on 4/5/17.
//

#pragma once

#include "../../framework/op_kernel.h"
#include "../../params/con_params.h"
#include "con_op_openblas.h"

namespace simpleCNN {

  class ConOp : public core::OpKernel {
   public:
    explicit ConOp(const core::OpKernelConstruction& context) : core::OpKernel(context) {}

    void compute(const core::OpKernelContext& context) override {
      core::Con_params* connected_params_ptr = static_cast<core::Con_params*>(core::OpKernel::params_);
      const auto& params                     = connected_params_ptr->connected_params();

      const tensor_t& in_data = context.input(0);
      const tensor_t& weight  = context.input(1);
      const tensor_t& bias    = context.input(2);
      tensor_t& out_data      = context.output(0);

      const core::backend_t engine = context.engine();

      if (engine == core::backend_t::internal) {
        kernels::con_op_openblas(in_data, weight, bias, out_data, params);
      } else {
        throw simple_error("No supported engine");
      }
    }
  };
}  // namespace simpleCNN
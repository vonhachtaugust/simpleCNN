//
// Created by hacht on 4/3/17.
//

#pragma once

#include "../framework/op_kernel.h"
#include "../params/max_params.h"
#include "max_op_internal.h"

namespace simpleCNN {
  class MaxpoolingOp : public core::OpKernel {
   public:
    explicit MaxpoolingOp(const core::OpKernelConstruction& context) : core::OpKernel(context) {}

    void compute(const core::OpKernelContext& context) override {
      /**
       *  Have a look at Conv2Op compute comment ...
       */
      core::Maxpooling_params* maxpooling_params_ptr = static_cast<core::Maxpooling_params*>(core::OpKernel::params_);
      const auto& params                             = maxpooling_params_ptr->maxpool();

      const tensor_t& in_data = context.input(0);
      tensor_t& out_data      = context.output(0);

      const core::backend_t engine = context.engine();

      if (engine == core::backend_t::internal) {
        kernels::maxpooling_op_internal(in_data, out_data, params);
      } else {
        throw simple_error("No supported engine");
      }
    }
  };
}  // namespace simpleCNN
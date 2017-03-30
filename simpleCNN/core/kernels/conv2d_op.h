//
// Created by hacht on 3/3/17.
//

#pragma once

#include "../params/conv_params.h"
#include "conv2d_op_openblas.h"

namespace simpleCNN {
  class Conv2Op : public core::OpKernel {
   public:
    explicit Conv2Op(const core::OpKernelConstruction& context) : core::OpKernel(context) {}

    void compute(const core::OpKernelContext& context) override {
      /** Questionable... Current situation: parameters shipped around
       * as shared_ptr<Params> which actually points to a specific
       * parameter type. Ideal would be to call params() on the
       * shared_ptr<Params> and let Params figure out which params
       * derived class is called. However, no elegant way of telling
       * params which derived class it is called on exists (usually
       * made by performing the call on the derived class instead of on
       * the base class).
       *
       * @breif Safely assures compiler that I know what I am doing (heh ...),
       * @details will return nullptr if core::OpKernel::params_.get()
       * isn't a core::Conv_params*
       */
      core::Conv_params* conv_params_ptr = static_cast<core::Conv_params*>(core::OpKernel::params_);
      const auto& params                 = conv_params_ptr->conv();

      // incoming/outcoming data
      const tensor_t& in_data = context.input(0);
      const tensor_t& weight  = context.input(1);
      const tensor_t& bias    = context.input(2);
      tensor_t& out_data      = context.output(0);

      const core::backend_t engine = context.engine();

      if (engine == core::backend_t::internal) {
        kernels::conv2d_op_openblas(in_data, weight, bias, out_data, params);
      } else {
        throw simple_error("No supported engine: ");
      }
    }
  };
}  // namespace simpleCNN

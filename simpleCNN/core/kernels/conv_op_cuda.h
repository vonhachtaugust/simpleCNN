//
// Created by hacht on 5/20/17.
//

#pragma once

#include "../params/conv_params.h"
#include "conv_op_openblas.h"

namespace simpleCNN {
class ConvCudaForwardOp : public core::OpKernel {
 public:
  explicit ConvCudaForwardOp(const core::OpKernelConstruction& context) : core::OpKernel(context) {}

  void compute(const core::OpKernelContext& context) override {
    // convolution params
    core::Conv_params* conv_params_ptr = static_cast<core::Conv_params*>(core::OpKernel::params_);
    const auto& params                 = conv_params_ptr->conv();

    // incoming/outcoming data
    const tensor_t& in_data = context.input(0);
    const tensor_t& weight  = context.input(1);
    const tensor_t& bias    = context.input(2);
    tensor_t& out_data      = context.output(0);









  }
};
}  // namespace simpleCNN



}
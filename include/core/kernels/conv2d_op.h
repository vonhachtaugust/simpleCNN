//
// Created by hacht on 3/3/17.
//

#pragma once

#include "../framework/op_kernel.h"
#include "../params/conv_params.h"
#include "conv2d_op_openblas.h"

namespace simpleCNN {
    class Conv2Op : public core::OpKernel {
    public:
        explicit Conv2Op(const core::OpKernelConstruction& context)
                :core::OpKernel(context) { }

        void compute(const core::OpKernelContext& context) override
        {
            auto params = core::OpKernel::params_->conv();

            // incoming/outcoming data
            const tensor_t& in_data = context.input(0);
            const tensor_t& weight = context.input(1);
            const tensor_t& bias = context.input(2);
            tensor_t& out_data = context.output(1);

            // initalize output
            out_data.fill(0.f);

            const core::backend_t engine = context.engine();

            if (engine==core::backend_t::internal)
            {
                kernels::conv2d_op_openblas(in_data, weight, bias, out_data, params);
            }
            else
            {
                throw simple_error("No supported engine: ");
            }
        }
    };
} // namespace simpleCNN

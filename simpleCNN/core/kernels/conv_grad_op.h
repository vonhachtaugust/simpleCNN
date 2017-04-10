//
// Created by hacht on 3/27/17.
//

#pragma once

namespace simpleCNN {

  class ConvGradOp : public core::GradOpKernel {
   public:
    explicit ConvGradOp(const core::OpKernelConstruction &context) : core::GradOpKernel(context) {}

    void compute(const core::OpKernelContext &context) override {
      core::Conv_params *conv_params_ptr = static_cast<core::Conv_params *>(core::GradOpKernel::params_);
      const auto &params                 = conv_params_ptr->conv();

      const tensor_t &weight                    = context.input(1);
      tensor_t &prev_delta                      = context.input_grad(0);
      tensor_t &curr_delta                      = context.output_grad(1);

      const core::backend_t engine = context.engine();
      if (engine == core::backend_t::internal) {
        kernels::conv_op_openblas(weight, prev_delta, curr_delta, params);
      } else {
        throw simple_error("No supported engine: ");
      }
    }

    /**
     * The purpose of this function is not to actually change any values but to prepare the optimizer for updating the weight by setting up dW and db.
     *
     * @param context
     */
    void update(const core::OpKernelContext &context) override {
      core::Conv_params *conv_params_ptr = static_cast<core::Conv_params *>(core::GradOpKernel::params_);
      const auto &params                 = conv_params_ptr->conv();

      const tensor_t& prev_in = context.input(0);
      const tensor_t& weight = context.input(1);
      tensor_t& dW = context.input_grad(1);
      tensor_t& db = context.input_grad(2);
      const tensor_t& curr_delta = context.output_grad(1);

      const core::backend_t engine = context.engine();

      if (engine == core::backend_t::internal) {
        kernels::conv_op_openblas(prev_in, weight, dW, db, curr_delta, params);
      } else {
        throw simple_error("No supported engine: ");
      }
    }
  };
}  // namespace simpleCNN

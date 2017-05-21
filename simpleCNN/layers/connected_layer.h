//
// Created by hacht on 4/5/17.
//

#pragma once

#include "../core/kernels/con_grad_op.h"
#include "../core/kernels/con_op.h"
#include "../core/params/con_params.h"
#include "layer.h"
#include "../core/kernels/con_op_cuda.h"

namespace simpleCNN {
  class Connected_layer : public Layer {
   public:
    Connected_layer(size_t in_dim,
                    size_t out_dim,
                    size_t batch_size,
                    bool has_bias                = true,
                    core::backend_t backend_type = core::default_engine())
      : Layer(std_input_order(has_bias), {tensor_t(component_t::OUT_DATA)}) {
      con_set_params(in_dim, out_dim, batch_size, has_bias);
      init_backend(backend_type);
      Layer::set_backend_type(backend_type);
      Layer::set_trainable(true);
    }

    size_t fan_in_size() const override { return params_.in_dim; }

    size_t fan_out_size() const override { return params_.out_dim; }

    shape_t in_shape() const override {
      if (params_.has_bias) {
        return {{params_.batch_size, 1, params_.in_dim, 1},
                {1, 1, params_.out_dim, params_.in_dim},
                {1, 1, params_.out_dim, 1}};
      }
      return {{params_.batch_size, 1, params_.in_dim, 1}, {1, 1, params_.out_dim, params_.in_dim}};
    }

    shape_t out_shape() const override { return {{params_.batch_size, 1, params_.out_dim, 1}}; }

    void forward_propagation(const data_ptrs_t& in_data, data_ptrs_t& out_data) override {
      // Reshape input to suit connected layer format.
      // printc(in_data[0]->shape(), "Init");

      auto ctx = core::OpKernelContext(in_data, out_data);
      ctx.setEngine(Layer::engine());

      // Reshape input to suit connected layer format.
      auto shape = in_data[0]->shape();
      Layer::reshape(*in_data[0], in_shape()[0]);

      kernel_fwd_->compute(ctx);

      // Reshape input back
      in_data[0]->reshape(shape);
    }

    void back_propagation(const data_ptrs_t& in_data,
                          const data_ptrs_t& out_data,
                          data_ptrs_t& in_grad,
                          data_ptrs_t& out_grad) override {
      auto ctx = core::OpKernelContext(in_data, out_data, in_grad, out_grad);
      ctx.setEngine(Layer::engine());

      auto shape = in_data[0]->shape();
      Layer::reshape(*in_data[0], in_shape()[0]);
      Layer::reshape(*in_grad[0], in_shape()[0]);

      kernel_bwd_->compute(ctx);
      in_data[0]->reshape(shape);
      in_grad[0]->reshape(shape);
    }

    tensor_t& network_output() override { return *Layer::out_component_data(component_t::OUT_DATA); };

    std::string layer_type() const override { return std::string("connected"); }

   private:
    void con_set_params(size_t in_dim, size_t out_dim, size_t batch_size, bool has_bias) {
      params_.in_dim     = in_dim;
      params_.out_dim    = out_dim;
      params_.batch_size = batch_size;
      params_.has_bias   = has_bias;
    }

    void init_backend(const core::backend_t backend_type) {
      core::OpKernelConstruction ctx(Layer::device(), &params_);

      if (backend_type == core::backend_t::internal) {
        kernel_fwd_.reset(new simpleCNN::ConOp(ctx));
        kernel_bwd_.reset(new simpleCNN::ConGradOp(ctx));
      } else if(backend_type == core::backend_t::gpu) {
        params_.initialize_gpu_descriptors();
        kernel_fwd_.reset(new simpleCNN::ConCudaForwardOp(ctx));
        kernel_bwd_.reset(new simpleCNN::ConCudaBackwardGradOp(ctx));
      } else {
        throw simple_error("No supported engine: ");
      }
    }
    /**
     * Set of connected layer parameters
     */
    core::Con_params params_;

    /**
     * Forward and backward ops
     */
    std::unique_ptr<core::OpKernel> kernel_fwd_;
    std::unique_ptr<core::OpKernel> kernel_bwd_;
  };
}  // namespace simpleCNN

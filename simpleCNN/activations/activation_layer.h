//
// Created by hacht on 5/8/17.
//

#pragma once

#include "../core/framework/op_kernel.h"
#include "../core/kernels/activation_kernels/activation_op.h"
#include "../core/kernels/activation_kernels/activation_op_cuda.h"
#include "../core/params/activation_params.h"
#include "../layers/layer.h"
#include "../util/util.h"

namespace simpleCNN {

  /**
   * Activation layer performing forward and backward activation of the affine transformation.
   * Input data holds the affine transformation and output data holds the activated values.
   */
  class Activation_layer : public Layer {
   public:
    Activation_layer(core::activation_t h         = core::activation_t::relu,
                     core::backend_t backend_type = core::default_engine())
      : Layer({tensor_t(component_t::IN_DATA)}, {tensor_t(component_t::OUT_DATA)}) {
      activation_set_params({0, 0, 0, 0}, h);
      Layer::set_backend_type(backend_type);
      Layer::set_trainable(false);
    }

    Activation_layer(shape4d shape,
                     core::activation_t h         = core::activation_t::relu,
                     core::backend_t backend_type = core::default_engine())
      : Layer({tensor_t(component_t::IN_DATA)}, {tensor_t(component_t::OUT_DATA)}) {
      activation_set_params(shape, h);
      init_backend(backend_type);
      Layer::set_backend_type(backend_type);
      Layer::set_trainable(false);
    }

    shape_t in_shape() const override { return {params_.shape}; }

    shape_t out_shape() const override { return {params_.shape}; }

    void set_in_shape(const shape4d& shape) override {
      params_.shape = shape;
    }

    void forward_propagation(const data_ptrs_t& in_data, data_ptrs_t& out_data) override {
      auto ctx = core::OpKernelContext(in_data, out_data);
      ctx.setEngine(Layer::engine());

      if (backend_initialized) {
        kernel_fwd_->compute(ctx);
      } else if (product(params_.shape) != 0) {
        init_backend(Layer::engine());
        kernel_fwd_->compute(ctx);
      } else {
          throw simple_error("No backend initialized");
      };
    }

    void back_propagation(const data_ptrs_t& in_data,
                          const data_ptrs_t& out_data,
                          data_ptrs_t& in_grad,
                          data_ptrs_t& out_grad) override {
      auto ctx = core::OpKernelContext(in_data, out_data, in_grad, out_grad);
      ctx.setEngine(Layer::engine());

      if (backend_initialized) {
        kernel_bwd_->compute(ctx);
      } else if (product(params_.shape) != 0) {
        init_backend(Layer::engine());
        kernel_bwd_->compute(ctx);
      } else {
        throw simple_error("No backend initizalied");
      }
    }

    std::string layer_type() const {
      switch (params_.activation_function) {
        case core::activation_t::relu: {
          return "relu";
        }
        case core::activation_t::softmax: {
          return "softmax";
        }
        case core::activation_t::tanh: {
          return "tanh";
        }
        default: {
          throw simple_error("No known activation function in use");
        }
      }
    }

   private:
    void activation_set_params(const shape4d& shape, const core::activation_t& activation_function) {
      params_.shape               = shape;
      params_.activation_function = activation_function;
    }

    void init_backend(const core::backend_t backend_type) {
      core::OpKernelConstruction ctx(Layer::device(), &params_);

      if (backend_type == core::backend_t::internal) {
        kernel_fwd_.reset(new simpleCNN::ActivationOp(ctx));
        kernel_bwd_.reset(new simpleCNN::ActivationGradOp(ctx));
      } else if (backend_type == core::backend_t::gpu) {
        kernel_fwd_.reset(new simpleCNN::ActivationCudaForwardOp(ctx));
        kernel_bwd_.reset(new simpleCNN::ActivationCudaBackwardOp(ctx));
      } else {

      }

      backend_initialized = true;
    }
    bool backend_initialized = false;

    /** Activation params */
    core::Activation_params params_;

    /** Forward and backward ops */
    std::unique_ptr<core::OpKernel> kernel_fwd_;
    std::unique_ptr<core::OpKernel> kernel_bwd_;
  };
}
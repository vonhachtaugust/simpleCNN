//
// Created by hacht on 4/26/17.
//

#pragma once

#include "../core/kernels/dropout_kernels/dropout_op.h"
#include "../core/kernels/dropout_kernels/dropout_op_cuda.h"
#include "../core/params/dropout_params.h"
#include "../network.h"

namespace simpleCNN {

  class Dropout_layer : public Layer {
   public:
    Dropout_layer(float_t prob,
                  core::backend_t backend_type = core::default_engine(),
                  net_phase phase              = net_phase::train)
      : Layer({tensor_t(component_t::IN_DATA)}, {tensor_t(component_t::OUT_DATA)}) {
      dropout_set_params(prob, phase);
      Layer::set_backend_type(backend_type);
      Layer::set_trainable(false);
    }

    shape_t in_shape() const override { return {params_.shape}; }

    shape_t out_shape() const override { return {params_.shape}; }

    /** Initialization is made through inferring the shape from the layer with this layer connects to */
    void set_in_shape(const shape4d& shape) override {
      params_.shape   = shape;
      params_.in_size = product(shape);
      params_.mask    = tensor_t(shape);
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
      }
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
        throw simple_error("No backend initialized");
      }
    }

    void set_netphase(net_phase phase) override { params_.phase = phase; }

    std::string layer_type() const override { return "dropout"; }

   private:
    void dropout_set_params(float_t prob, net_phase phase) {
      params_.prob  = prob;
      params_.phase = phase;
    }

    void init_backend(const core::backend_t backen_type) {
      core::OpKernelConstruction ctx(Layer::device(), &params_);

      if (backen_type == core::backend_t::internal) {
        kernel_fwd_.reset(new simpleCNN::DropoutOp(ctx));
        kernel_bwd_.reset(new simpleCNN::DropoutGradOp(ctx));
      } else if (backen_type == core::backend_t::gpu) {
        kernel_fwd_.reset(new simpleCNN::DropoutCudaForwardOp(ctx));
        kernel_bwd_.reset(new simpleCNN::DropoutCudaBackwardOp(ctx));
      } else {
        throw simple_error("No supported engine");
      }

      backend_initialized = true;
    }

    bool backend_initialized = false;

    /** Dropout params */
    core::Dropout_params params_;

    /** Forward and backward ops */
    std::unique_ptr<core::OpKernel> kernel_fwd_;
    std::unique_ptr<core::OpKernel> kernel_bwd_;
  };
}  // namespace simpleCNN

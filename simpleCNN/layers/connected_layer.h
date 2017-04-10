//
// Created by hacht on 4/5/17.
//

#pragma once

#include "../core/params/con_params.h"
#include "../core/kernels/con_op.h"
#include "../core/kernels/con_grad_op.h"
#include "layer.h"

namespace simpleCNN {

  template <typename T = float_t, typename Activation = activation::Identity<T>>
  class Connected_layer : public Feedforward_layer<T, Activation> {
   public:
    typedef Feedforward_layer<T, Activation> Base;
    Connected_layer(size_t in_dim,
                    size_t out_dim,
                    size_t batch_size,
                    bool has_bias                = true,
                    core::backend_t backend_type = core::default_engine())
      : Base(std_input_order(has_bias)) {
      con_set_params(in_dim, out_dim, batch_size, has_bias);
      init_backend(backend_type);
      Base::set_backend_type(backend_type);
    }

    size_t fan_in_size() const override { return params_.in_dim; }

    size_t fan_out_size() const override { return params_.out_dim; }

    shape_t in_shape() const override {
      if (params_.has_bias) {
      return {{params_.batch_size, 1, params_.in_dim, 1},
              {params_.batch_size, 1, params_.out_dim, params_.in_dim},
              {params_.batch_size, 1, params_.out_dim, 1}};
      }
      return {{params_.batch_size, 1, params_.in_dim, 1},
          {params_.batch_size, 1, params_.out_dim, params_.in_dim}};
    }

    shape_t out_shape() const override {
      return {{params_.batch_size, 1, params_.out_dim, 1}, {params_.batch_size, 1, params_.out_dim, 1}};
    }

    void forward_propagation(const data_ptrs_t& in_data, data_ptrs_t& out_data) override {
      auto ctx = core::OpKernelContext(in_data, out_data);
      ctx.setEngine(Layer::engine());
      ctx.setParams(&params_);

      kernel_fwd_->compute(ctx);
    }

    void back_propagation(const data_ptrs_t& in_data, const data_ptrs_t& out_data,
    data_ptrs_t& in_grad, data_ptrs_t& out_grad) override {
      auto ctx = core::OpKernelContext(in_data, out_data, in_grad, out_grad);
      ctx.setEngine(Layer::engine());
      ctx.setParams(&params_);

      kernel_bwd_->compute(ctx);
      kernel_bwd_->update(ctx);
    }

    std::string layer_type() const override { return std::string("connected"); }

    void createOp() override { init_backend(Layer::engine()); }

   private:
    void con_set_params(size_t in_dim,
                        size_t out_dim,
                        size_t batch_size,
                        bool has_bias) {
      params_.in_dim   = in_dim;
      params_.out_dim  = out_dim;
      params_.batch_size = batch_size;
      params_.has_bias = has_bias;
    }

    void init_backend(const core::backend_t backend_type) {
      core::OpKernelConstruction ctx(Layer::device(), &params_);

      if (backend_type == core::backend_t::internal) {
        kernel_fwd_.reset(new simpleCNN::ConOp(ctx));
        kernel_bwd_.reset(new simpleCNN::ConGradOp(ctx));
      } else {
        throw simple_error("No suppored engine: ");
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
    std::unique_ptr<core::GradOpKernel> kernel_bwd_;
  };
}  // namespace simpleCNN
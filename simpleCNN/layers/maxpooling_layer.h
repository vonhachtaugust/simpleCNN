//
// Created by hacht on 4/3/17.
//

#pragma once

#include "../activations/activation_function.h"
#include "../core/backend.h"
#include "../core/framework/op_kernel.h"
#include "../core/kernels/max_grad_op.h"
#include "../core/kernels/max_op.h"
#include "feedforward_layer.h"

namespace simpleCNN {

  template <typename T = float_t, typename Activation = activation::Identity<T>>
  class Maxpooling_layer : public Feedforward_layer<T, Activation> {
   public:
    typedef Feedforward_layer<T, Activation> Base;

    /**
     * Short version of constructing default maxpooling layer
     *
     */
    Maxpooling_layer(size_t in_width,
                     size_t in_height,
                     size_t in_channels,
                     size_t batch_size,
                     size_t pooling_size          = 2,
                     size_t stride                = 2,
                     core::backend_t backend_type = core::default_engine())
      : Maxpooling_layer(in_width,
                         in_height,
                         in_channels,
                         batch_size,
                         pooling_size,
                         pooling_size,
                         stride,
                         stride,
                         backend_type) {}

    /**
     * Constructing maxpooling layer
     *
     * @param in_width              input image width
     * @param in_height             input image height
     * @param in_channels           input image channels (grayscale = 1, rgb =
     * 3)
     * @param batch_size            number of input images to process
     * @param pooling_size_x        window width
     * @param pooling_size_y        window height
     * @param stride_x
     * @param stride_y
     * @param backend_type          specify backend engine to use
     */
    Maxpooling_layer(size_t in_width,
                     size_t in_height,
                     size_t in_channels,
                     size_t batch_size,
                     size_t pooling_size_x,
                     size_t pooling_size_y,
                     size_t stride_x,
                     size_t stride_y,
                     core::backend_t backend_type = core::default_engine())
      : Base({tensor_t(component_t::IN_DATA)}) {
      set_maxpooling_params(in_width, in_height, in_channels, batch_size,
                            pooling_size_x, pooling_size_y, stride_x, stride_y,
                            backend_type);
      init_backend(backend_type);
      Base::set_backend_type(backend_type);
    }

    void forward_propagation(const data_ptrs_t& in_data,
                             data_ptrs_t& out_data) override {
      auto ctx = core::OpKernelContext(in_data, out_data);
      ctx.setEngine(Layer::engine());
      ctx.setParams(&params_);

      kernel_fwd_->compute(ctx);
    }

    void back_propagation(const data_ptrs_t& in_data,
                          const data_ptrs_t& out_data,
                          data_ptrs_t& in_grad,
                          data_ptrs_t& out_grad) override {
      auto ctx = core::OpKernelContext(in_data, out_data, in_grad, out_grad);
      ctx.setEngine(Layer::engine());
      ctx.setParams(&params_);

      kernel_bwd_->compute(ctx);
      // Nothing to update.
    }

    shape_t in_shape() const override {
      return {{params_.batch_size, params_.in_channels, params_.input_height,
               params_.input_width}};
    }

    shape_t out_shape() const override {
      return {{params_.batch_size, params_.out_channels, params_.output_height,
               params_.output_width},
              {params_.batch_size, params_.out_channels, params_.output_height,
               params_.output_width}};
    }

    std::string layer_type() const override {
      return std::string("maxpooling");
    }

    void createOp() override { init_backend(Layer::engine()); }

   private:
    void set_maxpooling_params(size_t in_width,
                               size_t in_height,
                               size_t in_channels,
                               size_t batch_size,
                               size_t pooling_size_x,
                               size_t pooling_size_y,
                               size_t stride_x,
                               size_t stride_y,
                               core::backend_t = core::default_engine()) {
      params_.input_width  = in_width;
      params_.input_height = in_height;
      params_.in_channels  = in_channels;
      params_.batch_size   = batch_size;

      params_.pooling_size_x = pooling_size_x;
      params_.pooling_size_y = pooling_size_y;
      params_.stride_x       = stride_x;
      params_.stride_y       = stride_y;

      params_.output_width =
        params_.conv_out_length(in_width, pooling_size_x, stride_x, 0);
      params_.output_height =
        params_.conv_out_length(in_height, pooling_size_y, stride_y, 0);
      params_.out_channels = in_channels;

      params_.max_index = tensor_t(
        {batch_size, in_channels, params_.output_height, params_.output_width});
    }

    void init_backend(const core::backend_t backend_type) {
      core::OpKernelConstruction ctx(Layer::device(), &params_);

      if (backend_type == core::backend_t::internal) {
        kernel_fwd_.reset(new simpleCNN::MaxpoolingOp(ctx));
        kernel_bwd_.reset(new simpleCNN::MaxpoolingGradOp(ctx));
      } else {
        throw simple_error("No supported engine: ");
      }
    }

    /**
     * Set of maxpooling parameters
     **/
    core::Maxpooling_params params_;

    /**
   * @breif Forward and backward ops (only this object executes this kernel).
   *
   **/
    std::unique_ptr<core::OpKernel> kernel_fwd_;
    std::unique_ptr<core::OpKernel> kernel_bwd_;
  };
}

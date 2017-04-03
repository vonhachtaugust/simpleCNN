//
// Created by hacht on 3/1/17.
//

#pragma once

#include <algorithm>
#include <string>
#include <vector>

#include "feedforward_layer.h"

#include "../core/kernels/conv2d_op.h"
#include "../core/kernels/conv2d_grad_op.h"

namespace simpleCNN {

  template <typename T = float_t, typename Activation = activation::ReLU<T>>
  class Convolutional_layer : public Feedforward_layer<T, Activation> {
   public:
    typedef Feedforward_layer<T, Activation> Base;

    Convolutional_layer(size_t input_width,
                        size_t input_height,
                        size_t in_channels,
                        size_t batch_size,
                        size_t filter_size,
                        size_t out_channels,
                        size_t stride  = 1,
                        size_t padding = 0,
                        bool has_bias  = true)
      : Convolutional_layer(input_width,
                            input_height,
                            in_channels,
                            batch_size,
                            filter_size,
                            filter_size,
                            out_channels,
                            stride,
                            stride,
                            padding,
                            has_bias) {}

    /**
    * Constructing convolutional layer.
    *
    * @param input_width            [in] input image width
    * @param input_height           [in] input image height
    * @param in_channels            [in] input image channels (grayscale=1, rgb=3)
    * @param batch_size             [in] number of input images to processes in a forward pass
    * @param filter_width           [in] window_width(kernel) size of convolution
    * @param filter_height          [in] window_height(kernel) size of convolution
    * @param out_channels           [in] output image channels
    * @param horizontal_stride  [in] specify the horizontal interval at which to
    *apply the filters to the input
    * @param vertical_stride    [in] specify the vertical interval at which to
    *apply the filters to the input
    * @param padding                [in] number of paddings applied around the image
    * @param has_bias               [in] whether to add a bias vector to the filter outputs
     * @param backend_type       [in] specify backend engine you use
    **/
    Convolutional_layer(size_t input_width,
                        size_t input_height,
                        size_t in_channels,
                        size_t batch_size,
                        size_t filter_width,
                        size_t filter_height,
                        size_t out_channels,
                        size_t horizontal_stride,
                        size_t vertical_stride,
                        size_t padding,
                        bool has_bias,
                        core::backend_t backend_type = core::default_engine())
      : Base(std_input_order(has_bias)) {
      conv_set_params(input_width, input_height, in_channels, batch_size, filter_width, filter_height, out_channels,
                      horizontal_stride, vertical_stride, padding, has_bias);
      init_backend(backend_type);
      Base::set_backend_type(backend_type);
    }

    /**
     * @param in_data       input vector of this layer (data, weight, bias)
     * @param out_data      output vectors
     */
    void forward_propagation(const data_ptrs_t& in_data, data_ptrs_t& out_data) override {
      //data_ptrs_t in_data_(in_data);

      // forward convolution op context
      auto ctx = core::OpKernelContext(in_data, out_data);
      ctx.setEngine(Layer::engine());
      ctx.setParams(&params_);

      // launch convolutional kernel
      kernel_fwd_->compute(ctx);

      // this->forward_activation(*out_data[0],*out_data[1]);
    }

    void back_propagation(const data_ptrs_t & in_data,
        const data_ptrs_t & out_data,
                          data_ptrs_t & in_grad,
                          data_ptrs_t & out_grad) override {

      // backward convolution op context
      auto ctx = core::OpKernelContext(in_data, out_data, in_grad, out_grad);
      ctx.setEngine(Layer::engine());
      ctx.setParams(&params_);

      /*for (const auto& i : in_grad)
      {
        std::cout << *i << std::endl;
      }*/

      // launch convolutional kernel
      kernel_bwd_->compute(ctx);
    }


    shape_t in_shape() const override {
      if (params_.has_bias) {
        return {{params_.batch_size, params_.in_channels, params_.input_height, params_.input_width},
                {params_.out_channels, params_.in_channels, params_.filter_height, params_.filter_width},
                {params_.out_channels, 1, 1, 1}};
      } else {
        return {{params_.batch_size, params_.in_channels, params_.input_height, params_.input_width},
                {params_.out_channels, params_.in_channels, params_.filter_height, params_.filter_width}};
      }
    }

    shape_t out_shape() const override {
      return {{params_.batch_size, params_.out_channels, params_.output_height, params_.output_width},
              {params_.batch_size, params_.out_channels, params_.output_height, params_.output_width}};
    }

    std::string layer_type() const override { return std::string("conv"); }

    void createOp() override { init_backend(Layer::engine()); }

   private:
    /*
     * Default tensor_t has 4 dimensions where tensor_t(stack, depth, height, width)
     * according to row-major order: rightmost index varies fastest
     */
    void conv_set_params(size_t input_width,
                         size_t input_height,
                         size_t in_channels,
                         size_t batch_size,
                         size_t filter_width,
                         size_t filter_height,
                         size_t out_channels,
                         size_t horizontal_stride,
                         size_t vertical_stride,
                         size_t padding,
                         bool has_bias) {
      params_.input_width  = input_width;
      params_.input_height = input_height;
      params_.in_channels  = in_channels;
      params_.batch_size   = batch_size;

      params_.filter_width      = filter_width;
      params_.filter_height     = filter_height;
      params_.horizontal_stride = horizontal_stride;
      params_.vertical_stride   = vertical_stride;
      params_.padding           = padding;
      params_.has_bias          = has_bias;

      params_.output_width  = params_.conv_out_length(input_width, filter_width, vertical_stride, padding);
      params_.output_height = params_.conv_out_length(input_height, filter_height, horizontal_stride, padding);
      params_.out_channels  = out_channels;
    }

    void init_backend(const core::backend_t backend_type) {
      core::OpKernelConstruction ctx(Layer::device(), &params_);

      if (backend_type == core::backend_t::internal) {
        /*
         * RAII: C++ guarantees that the destructors of objects on the stack
         * will be called, even if an exception is thrown.
         */
        kernel_fwd_.reset(new simpleCNN::Conv2Op(ctx));
        kernel_bwd_.reset(new simpleCNN::Conv2dGradOp(ctx));
      } else {
        throw simple_error("No supported engine: ");
      }
    }

    /*
     * Set of convolutional parameters
     */
    core::Conv_params params_;

    /**
     * @breif Forward and backward ops (only this object executes this kernel).
     *
     * @note Alternative design to let each layer share ownership of
     * the one kernel. Since the data is sequentially propagated,
     * parallelism in execution is not possible.
     *
     **/
    std::unique_ptr<core::OpKernel> kernel_fwd_;
    std::unique_ptr<core::OpKernel> kernel_bwd_;
  };
}  // namespace simpleCNN

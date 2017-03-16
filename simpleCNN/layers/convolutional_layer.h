//
// Created by hacht on 3/1/17.
//

#pragma once

#include <algorithm>
#include <string>
#include <vector>

#include "../activations/activation_function.h"
#include "../core/framework/op_kernel.h"
#include "../core/kernels/conv2d_op.h"
#include "../core/params/conv_params.h"
#include "../util/util.h"

#include "feedforward_layer.h"

namespace simpleCNN {

  template <typename T = float_t, typename Activation = activation::ReLU<T>>
  class Convolutional_layer : public Feedforward_layer<T, Activation> {
   public:
    typedef Feedforward_layer<T, Activation> Base;

    Convolutional_layer(size_t in_width,
                        size_t in_height,
                        size_t filter_size,
                        size_t in_channels,
                        size_t out_channels,
                        size_t padding               = 0,
                        bool has_bias                = true,
                        size_t horizontal_stride     = 1,
                        size_t vertical_stride       = 1,
                        core::backend_t backend_type = core::default_engine())
      : Convolutional_layer(in_width,
                            in_height,
                            filter_size,
                            filter_size,
                            in_channels,
                            out_channels,
                            padding,
                            has_bias,
                            horizontal_stride,
                            vertical_stride) {}

    /**
    * Constructing convolutional layer.
    *
    * @param in_width           [in] input image width
    * @param in_height          [in] input image height
    * @param filter_width       [in] window_width(kernel) size of convolution
    * @param filter_height      [in] window_height(kernel) size of convolution
    * @param in_channels        [in] input image channels (grayscale=1, rgb=3)
    * @param out_channels       [in] output image channels
    * @param padding            [in] number of paddings applied around the image
    * @param has_bias           [in] whether to add a bias vector to the filter
    *outputs
    * @param horizontal_stride  [in] specify the horizontal interval at which to
    *apply the filters to the input
    * @param vertical_stride    [in] specify the vertical interval at which to
    *apply the filters to the input
    * @param backend_type       [in] specify backend engine you use
    **/
    Convolutional_layer(size_t in_width,
                        size_t in_height,
                        size_t filter_width,
                        size_t filter_height,
                        size_t in_channels,
                        size_t out_channels,
                        size_t padding,
                        bool has_bias,
                        size_t horizontal_stride,
                        size_t vertical_stride,
                        core::backend_t backend_type = core::default_engine())
      : Base(std_input_order(has_bias)) {
      conv_set_params(
        tensor_t({in_height, in_width, in_channels}, component_t::IN_DATA),
        filter_width, filter_height, out_channels, padding, has_bias,
        horizontal_stride, vertical_stride);
      init_backend(backend_type);
      Base::set_backend_type(backend_type);
    }

    /**
     * @param in_data       input vector of this layer (data, weight, bias)
     * @param out_data      output vectors
     */
    void forward_propagation(const vec_tensor_ptr_t& in_data,
                             vec_tensor_ptr_t& out_data) override {
      vec_tensor_ptr_t in_data_(in_data);

      // forward convolution op context
      auto ctx = core::OpKernelContext(in_data_, out_data);
      ctx.setEngine(Layer::engine());

      // launch convolutional kernel
      kernel_fwd_->compute(ctx);

      // this->forward_activation(*out_data[0],*out_data[1]);
    }

    data_t in_shape() const override {
      if (params_.has_bias) {
        return data_t(
          {params_.in, params_.weights,
           tensor_t({1, 1, params_.out.depth()}, component_t::BIAS)});
      } else {
        return data_t({params_.in, params_.weights});
      }
    }

    data_t out_shape() const override { return data_t({params_.out}); }

    std::string layer_type() const override { return std::string("conv"); }

    void createOp() override { init_backend(Layer::engine()); }

   private:
    /*
     * Default tensor_t has 3 dimensions where tensor_t(height, width, depth)
     * according to row-major order: rightmost index varies fastest
     */
    void conv_set_params(const tensor_t& in,
                         size_t filter_width,
                         size_t filter_height,
                         size_t out_channels,
                         size_t padding,
                         bool has_bias,
                         size_t horizontal_stride,
                         size_t vertical_stride) {
      params_.in = in;
      params_.out =
        tensor_t({params_.conv_out_length(in.height(), filter_height,
                                          vertical_stride, padding),
                  params_.conv_out_length(in.width(), filter_width,
                                          horizontal_stride, padding),
                  out_channels},
                 component_t::OUT_DATA);
      params_.weights =
        tensor_t({filter_height, filter_width, in.depth() * out_channels},
                 component_t::WEIGHT);
      params_.has_bias          = has_bias;
      params_.padding           = padding;
      params_.horizontal_stride = horizontal_stride;
      params_.vertical_stride   = vertical_stride;
    }

    void init_backend(const core::backend_t backend_type) {
      core::OpKernelConstruction ctx(Layer::device(), &params_);

      if (backend_type == core::backend_t::internal) {
        /*
         * RAII: C++ guarantees that the destructors of objects on the stack
         * will be called, even if an exception is thrown.
         */
        kernel_fwd_.reset(new simpleCNN::Conv2Op(ctx));
        //                kernel_bwd_.reset(new Conv2dGradOp(ctx));
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

//
// Created by hacht on 5/8/17.
//

#pragma once

#include "../layers/layer.h"
#include "../util/util.h"
#include "../../third_party/cudnn/include/cudnn.h"
#include "../util/cuda_utils.h"

namespace simpleCNN {

  /**
   * Activation layer performing forward and backward activation of the affine transformation.
   * Input data holds the affine transformation and output data holds the activated values.
   */
  class Activation_layer : public Layer {
   public:
    Activation_layer(core::backend_t backend_type = core::default_engine()) : Layer({tensor_t(component_t::IN_DATA)}, {tensor_t(component_t::OUT_DATA)}) {
      Layer::set_trainable(false);

      if (backend_type == core::backend_t::gpu) {
        initialize_gpu_descriptor();
      }
    }

    Activation_layer(shape4d shape, core::backend_t backend_type = core::default_engine()) : Layer({tensor_t(component_t::IN_DATA)}, {tensor_t(component_t::OUT_DATA)}) {
      shape_ = shape;
      Layer::set_trainable(false);

      if (backend_type == core::backend_t::gpu) {
        initialize_gpu_descriptor();
      }
    }


    shape_t in_shape() const override { return {shape_}; }

    shape_t out_shape() const override { return {shape_}; }

    void set_in_shape(const shape4d& shape) override {
      shape_ = shape;
    }

    void forward_propagation(const data_ptrs_t& in_data, data_ptrs_t& out_data) override {
      core::backend_t engine = Layer::engine();

      if (engine == core::backend_t::internal) {
        forward_activation(*in_data[0], *out_data[0]);
      } else if (engine == core::backend_t::gpu) {
        forward_activation_gpu(*in_data[0], *out_data[0]);
      } else {
        throw simple_error("Not a supported engine");
      }
    }

    void back_propagation(const data_ptrs_t& in_data,
                          const data_ptrs_t& out_data,
                          data_ptrs_t& in_grad,
                          data_ptrs_t& out_grad) override {
      core::backend_t engine = Layer::engine();

      if (engine == core::backend_t::internal) {
        backward_activation(*in_data[0], *out_grad[0], *in_grad[0]);
      } else if (engine == core::backend_t::gpu) {
        backward_activation_gpu(*in_data[0], *out_data[0], *out_grad[0], *in_grad[0]);
      } else {
        throw simple_error("Not a supported engine");
      }
    }

    virtual std::string layer_type() const = 0;

    /**
     * Forward pass consists of activating the input -> apply function to every value in tensor.
     *
     * @param affine          Affine transformation as outputted by previous layer.
     * @param activated       Activated affine transformation values forwarded to the next layer.
     */
    virtual void forward_activation(const tensor_t& affine, tensor_t& activated) const = 0;

    /**
     * Backward pass consists of scaling backward passing gradients/deltas by the
     * derivative of the activation function applied onto the affine transformations.
     *
     * @param affine          Affine transformation as outputted by previous layer.
     * @param curr_delta      Gradient/delta values passed backwards from the next layer.
     * @param activated       Scaled backward gradients by the derivative of the activated affine transformations.
     */
    virtual void backward_activation(const tensor_t& affine, const tensor_t& curr_delta, tensor_t& activated) const = 0;

    /** Only active if using gpu engine */
    virtual void forward_activation_gpu(const tensor_t& affine, tensor_t& activated) const = 0;

    /** Only active if using gpu engine */
    virtual void backward_activation_gpu(const tensor_t& affine, const tensor_t& activated, const tensor_t& curr_delta, tensor_t& prev_delta) const = 0;

    virtual std::pair<float_t, float_t> scale() const = 0;
   private:
    shape4d shape_;


   protected:
#ifdef USE_CUDNN
    cudnnHandle_t cudnnHandle;
    cudnnTensorDescriptor_t srcTensorDesc, dstTensorDesc;
    cudnnTensorDescriptor_t dsrcTensorDesc, ddstTensorDesc;
    cudnnActivationDescriptor_t Activation;
    bool initialized_gpu = false;

    void initialize_gpu_descriptor() {
      if (shape_.size() == 0) {
        throw simple_error("Activation layer shape not initialized");
      }

      cudnnHandle = cudnn_handle();
      checkCUDNN(cudnnCreateTensorDescriptor(&srcTensorDesc));
      checkCUDNN(cudnnSetTensor4dDescriptor(srcTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                            shape_[0], shape_[1],
                                            shape_[2], shape_[3]));
      checkCUDNN(cudnnCreateTensorDescriptor(&dsrcTensorDesc));
      checkCUDNN(cudnnSetTensor4dDescriptor(dsrcTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                            shape_[0], shape_[1],
                                            shape_[2], shape_[3]));

      checkCUDNN(cudnnCreateTensorDescriptor(&dstTensorDesc));
      checkCUDNN(cudnnSetTensor4dDescriptor(dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                            shape_[0], shape_[1],
                                            shape_[2], shape_[3]));

      checkCUDNN(cudnnCreateTensorDescriptor(&ddstTensorDesc));
      checkCUDNN(cudnnSetTensor4dDescriptor(ddstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                            shape_[0], shape_[1],
                                            shape_[2], shape_[3]));

      checkCUDNN(cudnnCreateActivationDescriptor(&Activation));
      checkCUDNN(cudnnSetActivationDescriptor(Activation,
                                              CUDNN_ACTIVATION_RELU,
                                              CUDNN_PROPAGATE_NAN,
                                              0.0));
      initialized_gpu = true;
    }

    ~Activation_layer() {
      if (initialized_gpu) {
        checkCUDNN(cudnnDestroyActivationDescriptor(Activation));
        checkCUDNN(cudnnDestroyTensorDescriptor(srcTensorDesc));
        checkCUDNN(cudnnDestroyTensorDescriptor(dstTensorDesc));
        checkCUDNN(cudnnDestroyTensorDescriptor(dsrcTensorDesc));
        checkCUDNN(cudnnDestroyTensorDescriptor(ddstTensorDesc));
      }
    }
#endif


  };
}
//
// Created by hacht on 5/8/17.
//

#pragma once

#include "../util/util.h"
#include "../layers/layer.h"

namespace simpleCNN {

/**
 * Activation layer performing forward and backward activation of the affine transformation.
 * Input data holds the affine transformation and output data holds the activated values.
 */
class Activation_layer : public Layer {
 public:
  Activation_layer() : Layer({tensor_t(component_t::IN_DATA)}, {tensor_t(component_t::OUT_DATA)}) {
    Layer::set_trainable(false);
  }

  Activation_layer(shape4d shape) : Layer({tensor_t(component_t::IN_DATA)}, {tensor_t(component_t::OUT_DATA)}) {
    shape_ = shape;
    Layer::set_trainable(false);
  }

  shape_t in_shape() const override { return {shape_}; }

  shape_t out_shape() const override { return {shape_}; }

  void set_in_shape(const shape4d& shape) override { shape_ = shape; }


  void forward_propagation(const data_ptrs_t& in_data, data_ptrs_t& out_data) override {
    forward_activation(*in_data[0], *out_data[0]);
  }

  void back_propagation(const data_ptrs_t& in_data, const data_ptrs_t& out_data, data_ptrs_t& in_grad, data_ptrs_t& out_grad) override {
    backward_activation(*in_data[0], *out_grad[0], *in_grad[0]);
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

  virtual std::pair<float_t, float_t> scale() const = 0;

 private:
  shape4d shape_;
};

}
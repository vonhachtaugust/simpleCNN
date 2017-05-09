//
// Created by hacht on 5/9/17.
//

#pragma once

namespace simpleCNN {

class Loss_layer : public Layer {
 public:
  Loss_layer() : Layer({tensor_t(component_t::IN_DATA)}, {tensor_t(component_t::OUT_DATA), tensor_t(component_t::TARGET)}) {
    Layer::set_trainable(false);
  }

  Loss_layer(shape4d shape) : shape_(shape), Layer({tensor_t(component_t::IN_DATA)}, {tensor_t(component_t::OUT_DATA), tensor_t(component_t::TARGET)}) {
    Layer::set_trainable(false);
  }

  virtual ~Loss_layer() = default;

  shape_t in_shape() const override { return {shape_}; }

  shape_t out_shape() const override { return {shape_, {shape_.front(), 1, 1, 1}}; }

  void set_in_shape(const shape4d& shape) override { shape_ = shape; }

  void set_targets(const tensor_t& labels) override { *Layer::out_component_data(component_t::TARGET) = labels; }

  /**
   * Applies the loss function onto the input data. (aka. output data of the network)
   *
   * @param in_data
   * @param out_data
   */
  void forward_propagation(const data_ptrs_t& in_data, data_ptrs_t& out_data) override {
    loss_function(*in_data[0], *out_data[0]);
  }

  /**
   * Applies the loss gradient function onto the values obtained by applying the loss function onto the input data
   * and prepares them for backward passes by putting the data as the input gradient.
   *
   * @param in_data
   * @param out_data
   * @param in_grad
   * @param out_grad
   */
  void back_propagation(const data_ptrs_t& in_data, const data_ptrs_t& out_data, data_ptrs_t& in_grad, data_ptrs_t& out_grad) override {
    loss_gradient(*out_data[0], *out_data[1], *in_grad[0]);
  }

  float_t error(const tensor_t& output, const tensor_t& target) const override {
    return loss(output, target);
  }

  /**
   * @return loss function applied to the network output data.
   */
  tensor_t network_output() override {
    return *Layer::out_component_data(component_t::OUT_DATA);
  };

  tensor_t network_target() override {
    return *Layer::out_component_data(component_t::TARGET);
  }

  /**
   * Function to compute network output into something interpretable, like probability distribution etc.
   *
   * @param in_data
   * @param out_data
   */
  virtual void loss_function(const tensor_t& in_data, tensor_t& out_data) const = 0;

  /**
   * Function to return the gradient with respect to the loss values.
   *
   * @param out_data
   * @param in_grad
   */
  virtual void loss_gradient(const tensor_t& out_data, const tensor_t& target, tensor_t& in_grad) const = 0;

  virtual float_t loss(const tensor_t& output, const tensor_t& target) const = 0;

  virtual std::string layer_type() const = 0;

 private:
  shape4d shape_;
};

}
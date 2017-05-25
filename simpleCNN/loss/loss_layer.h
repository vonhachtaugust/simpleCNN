//
// Created by hacht on 5/9/17.
//

#pragma once

namespace simpleCNN {

#include "../layers/layer.h"
#include "../util/util.h"

  class Loss_layer : public Layer {
   public:
    Loss_layer()
      : Layer({tensor_t(component_t::IN_DATA)}, {tensor_t(component_t::OUT_DATA), tensor_t(component_t::TARGET)}) {
      Layer::set_trainable(false);
    }

    Loss_layer(shape4d shape)
      : Layer({tensor_t(component_t::IN_DATA)}, {tensor_t(component_t::OUT_DATA), tensor_t(component_t::TARGET)}) {
      shape_ = shape;
      Layer::set_trainable(false);
    }

    shape_t in_shape() const override { return {shape_}; }

    shape_t out_shape() const override { return {shape_, {shape_[0], 1, 1, 1}}; }

    void set_in_shape(const shape4d& shape) override { shape_ = shape; }

    virtual void set_shape(const shape4d& shape) { shape_ = shape; }

    void set_targets(const tensor_t& labels) override {
      //*Layer::out_component_data(component_t::TARGET) = labels;
      Layer::set_out_data(labels, 1);
    }

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
    void back_propagation(const data_ptrs_t& in_data,
                          const data_ptrs_t& out_data,
                          data_ptrs_t& in_grad,
                          data_ptrs_t& out_grad) override {
      loss_gradient(*out_data[0], *out_data[1], *in_grad[0]);
    }

    float_t error(const tensor_t& output,
                  const tensor_t& target,
                  const std::vector<tensor_t*>& weights) const override {
      return loss(output, target);// + Hyperparameters::regularization_constant * regularization<float_t>(weights);
    }

    float_t accuracy(const tensor_t& output, const tensor_t& target) const override {
      size_t batch_size   = output.shape()[0];
      size_t batch_length = output.size() / batch_size;

      float_t acc = float_t(0);
      for (size_t b = 0; b < batch_size; ++b) {
        size_t max_index    = -1;
        float_t max         = float_t(-1);  // sometimes max is zero...
        size_t target_index = *(target.host_begin() + b);

        for (size_t j = 0; j < batch_length; ++j) {
          size_t index = b * batch_length + j;

          auto val = output.host_at_index(index);
          if (val > max) {
            max       = val;
            max_index = j;
          }
        }

        if (max_index == -1) {
          continue;
          // throw simple_error("Error: No max index was found");
        }

        if (max_index == target_index) {
          acc += float_t(1);
        }
      }
      // print(acc, "Accuracy");
      return acc / static_cast<float_t>(batch_size);
    }

    /**
     * @return loss function applied to the network output data.
     */
    tensor_t& network_output() override { return *Layer::out_component_data(component_t::OUT_DATA); };

    tensor_t& network_target() override { return *Layer::out_component_data(component_t::TARGET); }

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
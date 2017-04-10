//
// Created by hacht on 3/7/17.
//

#pragma once

#include "../activations/activation_function.h"
#include "layer.h"

namespace simpleCNN {
  template <typename T, typename Activation>
  class Feedforward_layer : public Layer {
   public:
    explicit Feedforward_layer(const data_t& in_data_type) : Layer(in_data_type, std_output_order(true)) {}

    activation::Function<T>& activation_function() { return h_; }

    std::pair<float_t, float_t> out_value_range() const override { return h_.scale(); };

    void forward_activation(const tensor_t& affine, tensor_t& activated) override {
      h_.a(affine, activated, affine.size());
    }

    void backward_activation(const tensor_t& affine, const tensor_t& prev_delta, tensor_t& activated) override {
      h_.da(affine, prev_delta, activated, affine.size());
    }

   private:
    Activation h_;
  };
}  // namespace simpleCNN

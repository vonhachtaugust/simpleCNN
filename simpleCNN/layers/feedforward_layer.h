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

    std::pair<T, T> out_value_range() const override { return h_.scale(); };

    void forward_activation() {}

    void backward_activation() {}

   private:
    Activation h_;
  };
}  // namespace simpleCNN

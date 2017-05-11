//
// Created by hacht on 5/8/17.
//

#pragma once

namespace simpleCNN {
  namespace activation {
  class ReLU : public Activation_layer {
   public:
    std::string layer_type() const override { return "ReLU-activation-layer"; }

    template<typename T>
    T f(const T& value) const { return std::max(T{0}, value); }

    template<typename T>
    T df(T value) const { return value > T{0} ? T{1} : T{0}; }

    void forward_activation(const tensor_t& affine, tensor_t& activated) const override {
      for (size_t i = 0; i < affine.size(); ++i) {
        activated.host_at_index(i) = f(affine.host_at_index(i));
      }
    }

    void backward_activation(const tensor_t& affine, const tensor_t& curr_delta, tensor_t& activated) const override {
      for (size_t i = 0; i < affine.size(); ++i) {
        activated.host_at_index(i) = df(affine.host_at_index(i)) * curr_delta.host_at_index(i);
      }
    };

    std::pair<float_t, float_t> scale() const override {
      return std::make_pair(float_t(0), float_t(1));
    };
  };
  } // namespace activation
} // namespace simpleCNN

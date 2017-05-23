//
// Created by hacht on 5/15/17.
//

#pragma once

namespace simpleCNN {
  namespace activation {
    class Tanh : public Activation_layer {
     public:
      std::string layer_type() const override { return "tanh-activation-layer"; }

      template <typename T>
      T f(const T& value) const {
        return std::tanh(value);
      }

      template <typename T>
      T df(T value) const {
        return (float_t(1) - f(value) * f(value));
      }

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

      void forward_activation_gpu(const tensor_t& affine, tensor_t& activated) const override{};

      void backward_activation_gpu(const tensor_t& affine,
                                   const tensor_t& activated,
                                   const tensor_t& curr_delta,
                                   tensor_t& prev_delta) const override{};

      std::pair<float_t, float_t> scale() const override { return std::make_pair(float_t(-1), float_t(1)); };
    };
  }  // namespace activation
}  // namespace simpleCNN

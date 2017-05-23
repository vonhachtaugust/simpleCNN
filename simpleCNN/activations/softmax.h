//
// Created by hacht on 5/8/17.
//

#pragma once

namespace simpleCNN {
  namespace activation {
    class Softmax : public Activation_layer {
     public:
      std::string layer_type() const override { return "softmax-activation-layer"; }

      void forward_activation(const tensor_t& affine, tensor_t& activated) const override {
        size_t batch_size   = affine.shape()[0];
        size_t batch_length = affine.size() / batch_size;

        /** For each tensor in the batch */
        for (size_t i = 0; i < batch_size; ++i) {
          size_t start_index = i * batch_length;

          /** Get numerical stabilizer */
          auto start = affine.host_begin() + start_index;
          auto end   = start + batch_length;
          auto ns    = *std::max_element(start, end);

          /** Get normalization constant */
          float_t sum = float_t(0);
          for (size_t j = 0; j < batch_length; ++j) {
            auto val = affine.host_at_index(start_index + j);
            sum += std::exp(val - ns);
          }

          /** Compute softmax probability */
          for (size_t n = 0; n < batch_length; ++n) {
            auto val                                 = affine.host_at_index(start_index + n);
            activated.host_at_index(start_index + n) = std::exp(val - ns) / sum;
          }
        }
      }

      void backward_activation(const tensor_t& affine, const tensor_t& curr_delta, tensor_t& activated) const override {
        throw simple_not_implemented_error();
      }

      void forward_activation_gpu(const tensor_t& affine, tensor_t& activated) const override{};

      void backward_activation_gpu(const tensor_t& affine,
                                   const tensor_t& activated,
                                   const tensor_t& curr_delta,
                                   tensor_t& prev_delta) const override{};

      std::pair<float_t, float_t> scale() const override { return std::make_pair(float_t(0), float_t(1)); };
    };
  }  // namespace activation
}  // namespace simpleCNN

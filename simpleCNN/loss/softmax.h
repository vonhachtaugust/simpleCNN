//
// Created by hacht on 5/9/17.
//

#pragma once

namespace simpleCNN {
  namespace loss {
    class Softmax : public Loss_layer {
     public:
      std::string layer_type() const override { return "Softmax-loss-layer"; }

      template <typename T>
      T df(T value) const {
        return value - T(1);
      }

      template <typename T>
      T f(const T &value) const {
        if (value == 0) {
          throw simple_error("Value equals zero, loss is infinity");
        }
        return -std::log(value);
      }

      void loss_function(const tensor_t &in_data, tensor_t &out_data) const override {
        size_t batch_size   = in_data.shape()[0];
        size_t batch_length = in_data.size() / batch_size;

        /** For each tensor in the batch */
        for (size_t i = 0; i < batch_size; ++i) {
          size_t start_index = i * batch_length;

          /** Get numerical stabilizer */
          auto start = in_data.host_begin() + start_index;
          auto end   = start + batch_length;
          auto ns    = *std::max_element(start, end);

          /** Get normalization constant */
          float_t sum = float_t(0);
          for (size_t j = 0; j < batch_length; ++j) {
            auto val = in_data.host_at_index(start_index + j);
            sum += std::exp(val - ns);
          }

          /** Compute softmax probabilities */
          for (size_t n = 0; n < batch_length; ++n) {
            auto val                                = in_data.host_at_index(start_index + n);
            out_data.host_at_index(start_index + n) = std::exp(val - ns) / sum;
          }
        }
      }

      void loss_gradient(const tensor_t &out_data, const tensor_t &target, tensor_t &in_grad) const override {
        size_t batch_size = out_data.shape()[0];
        size_t n          = out_data.size() / batch_size;
        //print(out_data, "Outdata");
        //print(target, "Targets");
        //size_t val = *target.host_begin();

        for (size_t b = 0; b < batch_size; ++b) {
          //float_t t = target.host_at_index(b);
          size_t t = *(target.host_begin() + b);
          //print(t, "Target");
          for (size_t i = 0; i < n; ++i) {
            size_t index = b * n + i;
            if (i == t) {
              in_grad.host_at_index(index) = df(out_data.host_at_index(index));
              continue;
            }
            in_grad.host_at_index(index) = out_data.host_at_index(index);
          }
        }
        //print(in_grad, "Input gradient");
      }

      float_t loss(const tensor_t &output, const tensor_t &target) const override {
        float_t loss_tot  = float_t(0);
        size_t batch_size = output.shape()[0];
        size_t n          = output.size() / batch_size;
        //print(target, "Target");
        //print(output, "Output");

        for (size_t b = 0; b < batch_size; ++b) {
          //size_t target_i = target.host_at_index(b);
          size_t target_i = *(target.host_begin() + b);
          size_t index    = b * n + target_i;

          loss_tot += f(output.host_at_index(index));
        }
        return loss_tot / static_cast<float_t>(batch_size);
      }
    };
  }  // namespace loss
}  // namespace simpleCNN

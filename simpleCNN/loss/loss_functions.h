//
// Created by hacht on 4/12/17.
//

#pragma once

#include "../core/framework/tensor_utils.h"
#include "../util/util.h"

namespace simpleCNN {
  namespace loss {

    /**
     * Regularization term. Without it solution is not guaranteed to be unique.
     *
     * @tparam T
     * @param weights
     * @return
     */
    template <typename T>
    T regularization(const tensor_t &weight) {
      return dot(weight, &(*weight.host_begin()), weight, &(*weight.host_begin()));
    }

    template <typename T = float_t>
    class Softmax_classifier {
     public:
      /**
       * Loss value for an individual training example i
       *
       * @param activated
       * @param target_index
       * @return
       */
      static T f(const T &value) { return -std::log(value); }

      /**
       * Gradient value for an individual training example i
       *
       * @param activated           : Activated (e.g. softmax) values at the output layer.
       * @param deltas              : Output error to be returned.
       * @param target_index        : Which class this example corresponded to.
       * @params n                  : Number of values to apply df onto (typical; width x height x depth).
       */
      static T df(T value) { return value - T(1); }

      static T L(const tensor_t &output, const tensor_t &target, const size_t batch_size) {
        T loss_i = T(0);
        size_t n = output.size() / output.shape()[0];

        for (size_t b = 0; b < batch_size; ++b) {
          size_t target_index = target.host_at_index(b);
          loss_i += f(output.host_at_index(b * n + target_index));
        }
        return loss_i;
      }

      static tensor_t dL(const tensor_t &output, const tensor_t &target, const size_t batch_size) {
        tensor_t delta(output.shape_v());

        size_t n = output.size() / output.shape()[0];
        for (size_t b = 0; b < batch_size; ++b) {
          size_t t = target.host_at_index(b * n);
          for (size_t i = 0; i < n; ++i) {
            if (i == t) {
              delta.host_at_index(b * n + i) = df(output.host_at_index(b * n + i));
              continue;
            }
            delta.host_at_index(b * n + i) = output.host_at_index(b * n + i);
          }
        }
        return delta;
      }

     private:
      Softmax_classifier() {}
    };

    template <typename T>
    static T df(T value) {
      return value - T(1);
    }

    template <typename T>
    static T f(const T &value) {
      return -std::log(value);
    }

    static tensor_t loss_function(const tensor_t &in_data) {
      tensor_t out_data(in_data.shape_v());
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
      return out_data;
    }

    static tensor_t loss_gradient(const tensor_t &out_data, const tensor_t &target) {
      tensor_t out_gradident(out_data.shape_v());
      size_t batch_size = out_data.shape()[0];
      size_t n          = out_data.size() / batch_size;

      for (size_t b = 0; b < batch_size; ++b) {
        size_t t = target.host_at_index(b * n);
        for (size_t i = 0; i < n; ++i) {
          if (i == t) {
            out_gradident.host_at_index(b * n + i) = loss::df(out_data.host_at_index(b * n + i));
            continue;
          }
          out_gradident.host_at_index(b * n + i) = out_data.host_at_index(b * n + i);
        }
      }
      return out_gradident;
    }

    static float_t loss(const tensor_t &output, const tensor_t &target) {
      float_t loss_tot  = float_t(0);
      size_t batch_size = output.shape()[0];
      size_t n          = output.size() / batch_size;

      for (size_t b = 0; b < batch_size; ++b) {
        size_t target_i = target.host_at_index(b);
        size_t index    = b * n + target_i;
        auto val        = output.host_at_index(index);
        auto val2       = loss::f(val);

        // loss_tot += loss::f(output.host_at_index(index));
        loss_tot += val2;
      }
      return loss_tot / static_cast<float_t>(batch_size);
    }
  }  // namespace loss

  tensor_t grads(const tensor_t &output, const tensor_t &targets) {
    auto val = loss::loss_function(output);
    return loss::loss_gradient(val, targets);
  }

  /*
  float_t accuracy(const tensor_t &prob_dist, const tensor_t &targets) {
    size_t batch_size   = prob_dist.shape()[0];
    size_t batch_length = prob_dist.size() / batch_size;

    float_t acc = float_t(0);
    for (size_t i = 0; i < batch_size; ++i) {
      size_t max_index    = -1;
      float_t max         = float_t(0);
      size_t target_index = targets.host_at_index(i);

      for (size_t j = 0; j < batch_length; ++j) {
        size_t index = i * batch_length + j;

        auto val = prob_dist.host_at_index(index);
        if (val > max) {
          max       = val;
          max_index = j;
        }
      }

      if (max_index == -1) {
        throw simple_error("Error: No max index was found");
      }

      if (max_index == target_index) {
        acc += float_t(1);
      }
    }
    return acc / static_cast<float_t>(batch_size);
  }
   */

  void loss_value(const tensor_t &output, const tensor_t &targets) {
    //auto val = loss::loss_function(output);
    //auto acc = accuracy(val, targets);
    //auto l   = loss::loss(val, targets);
    //std::cout << "Loss: " << l << " ; "
    //          << "Acc: " << acc << std::endl;
  }

  /**
   * Returns output layer deltas
   *
   * @tparam Loss           : Loss function type.
   * @param output_t        : Non-activated data from output layer.
   * @param target_t        : Tensor of target indices.
   * @return deltas that initiates backpropagation.
   */
  template <typename Loss>
  tensor_t gradient(const tensor_t &output, const tensor_t &target, const size_t batch_size) {
    return Loss::dL(output, target, batch_size);
  }

  template <typename Loss>
  float_t error(const tensor_t &output, const tensor_t &target, const size_t batch_size) {
    return Loss::L(output, target, batch_size);
  }

}  // namespace simpleCNN

//
// Created by hacht on 4/12/17.
//

#pragma once

#include "../util/util.h"
#include "../core/framework/tensor_utils.h"

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
    T regularization(const tensor_t& weight) {
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
      static T f(const T& value) { return -std::log(value); }

      /**
       * Gradient value for an individual training example i
       *
       * @param activated           : Activated (e.g. softmax) values at the output layer.
       * @param deltas              : Output error to be returned.
       * @param target_index        : Which class this example corresponded to.
       * @params n                  : Number of values to apply df onto (typical; width x height x depth).
       */
      static T df(T value) { return value - T(1); }

      static T L(const tensor_t& output, const tensor_t& target, const size_t batch_size) {
        T loss_i  = T(0);
        size_t n  = output.size() / output.shape()[0];

        for (size_t b = 0; b < batch_size; ++b) {
          size_t target_index = target.host_at_index(b);
          auto val = f(output.host_at_index(b * n + target_index));
          loss_i += val;
        }
        return loss_i;
      }

      static tensor_t dL(const tensor_t&output, const tensor_t& target, const size_t batch_size) {
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
  }  // namespace loss

  /**
   * Returns output layer deltas
   *
   * @tparam Loss           : Loss function type.
   * @param output_t        : Non-activated data from output layer.
   * @param target_t        : Tensor of target indices.
   * @return deltas that initiates backpropagation.
   */
  template<typename Loss>
  tensor_t gradient(const tensor_t& output, const tensor_t& target, const size_t batch_size) {
    return Loss::dL(output, target, batch_size);
  }

  template<typename Loss>
  float_t error(const tensor_t& output, const tensor_t& target, const size_t batch_size) {
    return Loss::L(output, target, batch_size);
  }

}  // namespace simpleCNN

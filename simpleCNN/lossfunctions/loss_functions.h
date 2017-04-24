//
// Created by hacht on 4/12/17.
//

#pragma once

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
    class Log_likelihood {
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

      static T loss(const tensor_t& output, const tensor_t& target, const size_t batch_size) {
        T loss_i  = T{0};
        size_t tn = target.size() / target.shape()[0];
        size_t n  = output.size() / output.shape()[0];

        for (size_t b = 0; b < batch_size; ++b) {
          size_t t = target.host_index(b * tn);
          loss_i += f(output.host_index(b * n + t));
        }
        return loss_i;
      }

      static void dL(const tensor_t& output, const tensor_t& target, tensor_t& delta, const size_t batch_size) {
        size_t n = output.size() / output.shape()[0];
        for (size_t b = 0; b < batch_size; ++b) {
          size_t t = target.host_index(b * n);
          for (size_t i = 0; i < n; ++i) {
            if (i == t) {
              delta.host_index(b * n + i) = df(output.host_index(b * n + i));
              continue;
            }
            delta.host_index(b * n + i) = output.host_index(b * n + i);
          }
        }
      }

     private:
      Log_likelihood() {}
    };

    /**
     * Deep Learning using Linear Support Vector Machines : https://arxiv.org/pdf/1306.0239.pdf
     *
     * @note Also known as multiclass SVM
     *
     * @tparam T - precision
     */
    template <typename T>
    class L2SVM {
     public:
      static T f(const T& value) { return std::max(0, value) * std::max(0, value); }

      static T df(T value) { simple_not_implemented_error(); }

      static T loss(const tensor_t& output, const size_t target_index) {
        /*T loss_i = T(0);

        auto activated_i = activated.host_begin();
        T target_activation = *(activated_i + target_index);
        T delta = T(1);

        for (size_t j = 0; j < n; ++j) {
          if (j == target_index) {
            continue;
          }
          loss_i += f(*activated_i++ - target_activation + delta);
        }*/
      }

     private:
      L2SVM() {}
    };
  }  // namespace loss

  /**
   * Returns output layer deltas
   *
   * @tparam Loss           : Type of loss function.
   * @param output_t        : Activated (e.g. softmax) data from output layer.
   * @param target_t        : List of correct targets for this example.
   * @param target_loss_t   : The computed loss for each example in the batch.
   * @return deltas that initiates backpropagation.
   */
  template <typename Loss>
  void gradient(const tensor_t& output, const tensor_t& target, tensor_t& output_delta, const size_t batch_size) {
    Loss::dL(output, target, output_delta, batch_size);
  }
}  // namespace simpleCNN

//
// Created by hacht on 3/7/17.
//

#pragma once

#include <algorithm>
#include <utility>
#include "../util/util.h"

namespace simpleCNN {
  namespace activation {
    /*
    template <typename T = float_t>
    class Activation_Function {
     public:
      virtual ~Activation_Function() = default;
      virtual T f(const T& v) const  = 0;
      virtual T df(T y) const        = 0;

      void activate(const tensor_t& affine, tensor_t& activated) const {
        auto affine_i    = affine.host_begin();
        auto activated_i = activated.host_begin();

        for (size_t i = 0; i < affine.size(); ++i) {
          *activated_i++ = f(*affine_i++);
        }
      }

      void dactivate(const tensor_t& affine, const tensor_t& curr_delta, tensor_t& activated) const {
        auto curr_delta_i = curr_delta.host_begin();
        auto affine_i     = affine.host_begin();
        auto activated_i  = activated.host_begin();

        for (size_t i = 0; i < affine.size(); ++i) {
          *activated_i++ = df(*affine_i++) * *curr_delta_i++;
        }
      }

      // target value range for learning
      // virtual std::pair<float_t, float_t> scale() const = 0;
    };

    template <typename T = float_t>
    class ReLU : public Activation_Function<T> {
     public:
      T f(const T& v) const override { return std::max(T{0}, v); }
      T df(T y) const override { return y > T{0} ? T{1} : T{0}; }
    };

    template <typename T = float_t>
    class Identity : public Activation_Function<T> {
     public:
      T f(const T& v) const override { return v; }
      T df(T y) const override { return T(1); }
    };

    template <typename T = float_t>
    class Softmax : public Activation_Function<T> {
     public:
      T f(const T& v) const override { return std::exp(v); }
      T df(T y) const override { return T(0); }

      T f(const T& v, const T& numerical_stabilizer) const { return std::exp(v - numerical_stabilizer); }

      /**
       * Hides Function 'activate' since it needs to divide with something
       * that is determined by the whole data structure and cannot be deduced from one element.
       *
       * Also ignores f and uses it's own in order to deal with numerical problems.
       *
       * Not implementing df since it is supposed to be used as output layer, df cancels
       * in the loss calculation, that is the point ...
       *
       * @param affine
       * @param activated
       * @param n
       */
    /*
      void activate(const tensor_t& affine, tensor_t& activated) const {
        size_t batch_size = affine.shape()[0];
        size_t batch_length = affine.size() / batch_size;

        for (size_t i = 0; i < batch_size; ++i) {
          size_t start_index = i * batch_length;

          auto start = affine.host_begin() + start_index;
          auto end = affine.host_begin() + start_index + batch_length;

          T ns = *std::max_element(start, end);
          T denom = T(0);

          for (size_t j = 0; j < batch_length; ++j) {
            denom += f(affine.host_at_index(start_index + j), ns);
          }

          for (size_t n = 0; n < batch_length; ++n) {
            activated.host_at_index(start_index + n) = f(affine.host_at_index(start_index + n), ns) / denom;
          }
        }
      }
    };
    */
  }  // namespace activation
}  // namespace simpleCNN

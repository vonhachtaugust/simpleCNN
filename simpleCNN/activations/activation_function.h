//
// Created by hacht on 3/7/17.
//

#pragma once

#include <algorithm>
#include <utility>
#include "../util/util.h"

namespace simpleCNN {
  namespace activation {
    template <typename T = float_t>
    class Activation_Function {
     public:
      virtual ~Activation_Function() = default;
      virtual T f(const T& v) const  = 0;
      virtual T df(T y) const        = 0;

      void activate(const tensor_t& affine, tensor_t& activated, const size_t n) const {
        auto affine_i    = affine.host_begin();
        auto activated_i = activated.host_begin();

        for (size_t i = 0; i < n; ++i) {
          *activated_i++ = f(*affine_i++);
        }
      }

      void dactivate(const tensor_t& affine, const tensor_t& curr_delta, tensor_t& activated, const size_t n) const {
        auto curr_delta_i = curr_delta.host_begin();
        auto affine_i     = affine.host_begin();
        auto activated_i  = activated.host_begin();

        for (size_t i = 0; i < n; ++i) {
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
      void activate(const tensor_t& affine, tensor_t& activated, const size_t n) const {
        T numerical_stabilizer = *std::max_element(affine.host_begin(), affine.host_end());
        T denom                = T(0);

        auto affine_i = affine.host_begin();
        for (size_t j = 0; j < n; ++j) {
          denom += f(*affine_i++, numerical_stabilizer);
        }

        affine_i         = affine.host_begin();
        auto activated_i = activated.host_begin();

        for (size_t i = 0; i < n; ++i) {
          *activated_i++ = f(*affine_i++, numerical_stabilizer) / denom;
        }
      }
    };
  }  // namespace activation
}  // namespace simpleCNN

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
    class Function {
     public:
      Function() = default;
      Function(const Function&) = default;
      Function& operator=(const Function&) = default;

      /**
       * Virtuals
       */
      virtual ~Function() = default;

      /**
       * Pure virtuals
       */
      virtual T f(const T& v) const = 0;
      virtual T df(T y) const = 0;


      void a(const tensor_t& affine, tensor_t& activated, const size_t n) const {
        auto affine_i = affine.host_begin();
        auto activated_i = activated.host_begin();

        for (size_t i = 0; i < n; ++i) { *activated_i++ = f(*affine_i++); }
      }

      void da(const tensor_t& affine, const tensor_t& prev_delta, tensor_t& activated, const size_t n) const {
        auto prev_delta_i = prev_delta.host_begin();
        auto affine_i = affine.host_begin();
        auto activated_i = activated.host_begin();

        for (size_t i = 0; i < n; ++i) { *activated_i++ = df(*affine_i++) * *prev_delta_i++; }
      }

      // return if dfi/dyk is one-hot vector
      virtual bool one_hot() const { return true; }

      // target value range for learning
      virtual std::pair<float_t, float_t> scale() const = 0;
    };

    template <typename T = float_t>
    class ReLU : public Function<T> {
     public:
      T f(const T& v) const override { return std::max(T{0}, v); }

      T df(T y) const override { return y > T{0} ? T{1} : T{0}; }

      std::pair<float_t, float_t> scale() const override { return std::make_pair(float_t(0.1), float_t(0.9)); }
    };

    template<typename T = float_t>
    class Identity : public Function<T>
    {
     public:
      T f(const T& v) const override { return v; }

      T df(T y) const override { return T(0); }

      std::pair<float_t, float_t> scale() const override { return std::make_pair(float_t(0.1), float_t(0.9)); }
    };
  }  // namespace activation
}  // namespace simpleCNN

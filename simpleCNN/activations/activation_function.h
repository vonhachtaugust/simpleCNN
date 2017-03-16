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

      virtual ~Function() = default;

      virtual T f(const vec_t& v, size_t index) const = 0;

      virtual T df(T y) const = 0;

      void itef(vec_t& out, const vec_t& in, size_t cnt) const {
        for (size_t i = 0; i < cnt; ++i) {
          out[i] = f(in, i);
        }
      }

      void itedf(vec_t& cur,
                 const vec_t& prev,
                 const vec_t& out,
                 size_t cnt) const {
        for (size_t i = 0; i < cnt; ++i) {
          cur[i] = prev[i] * df(out[i]);
        }
      }

      // dfi/dyk (k=0,1,..n)
      virtual vec_t df(const vec_t& y, size_t i) const {
        vec_t v(y.size(), 0);
        v[i] = df(y[i]);
        return v;
      }

      // return if dfi/dyk is one-hot vector
      virtual bool one_hot() const { return true; }

      // target value range for learning
      virtual std::pair<T, T> scale() const = 0;
    };

    template <typename T = float_t>
    class ReLU : public Function<T> {
     public:
      T f(const vec_t& v, size_t i) const override {
        return std::max(T{0}, v[i]);
      }

      T df(T y) const override { return y > T{0} ? T{1} : T{0}; }

      std::pair<T, T> scale() const override {
        return std::make_pair(float_t(0.1), float_t(0.9));
      }
    };

  }  // namespace activation
}  // namespace simpleCNN

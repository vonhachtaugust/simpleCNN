//
// Created by hacht on 3/21/17.
//

#pragma once

#include "../core/framework/tensor_utils.h"
#include "random.h"
#include "util.h"

namespace simpleCNN {
  namespace weight_init {

    class Function {
     public:
      virtual void fill(tensor_t* weight, size_t fan_in, size_t fan_out) = 0;
    };

    class Scalable : public Function {
     public:
      explicit Scalable(float_t value) : scale_(value) {}

      void scale(float_t value) { scale_ = value; }

     protected:
      float_t scale_;
    };

    class Gaussian : public Scalable {
     public:
      Gaussian() : Scalable(float_t(2)) {}
      explicit Gaussian(float_t value) : Scalable(value) {}

      void fill(tensor_t* weight, size_t fan_in, size_t fan_out) override {
        // Average version: const float_t std = std::sqrt(2 * scale_ / (fan_in + fan_out));
        const float_t std = std::sqrt(scale_ / fan_in);

        normal_dist(weight->host_begin(), weight->host_end(), float_t(0), std);
      }
    };

    // Horrible for ReLU
    class Xavier : public Scalable {
     public:
      Xavier() : Scalable(float_t(6)) {}
      explicit Xavier(float_t value) : Scalable(value) {}

      void fill(tensor_t* weight, size_t fan_in, size_t fan_out) override {
        // Average version: const float_t weight_base = std::sqrt(scale_ / (fan_in + fan_out));
        // Fan in version
        const float_t weight_base = std::sqrt(3 / (fan_in));

        uniform_rand(weight->host_begin(), weight->host_end(), -weight_base, weight_base);
      }
    };

    class Constant : public Scalable {
     public:
      Constant() : Scalable(float_t(0)) {}
      explicit Constant(float_t value) : Scalable(value) {}

      void fill(tensor_t* weight, size_t fan_in, size_t fan_out) override {
        std::fill(weight->host_begin(), weight->host_end(), scale_);
      }
    };

    class Test : public Scalable {
     public:
      // Test() : Scalable(float_t(6)) {}
      explicit Test(vec_t& data) : Scalable(1.0f), data_(data) {}

      void fill(tensor_t* weight, size_t fan_in, size_t fan_out) override { simpleCNN::fill(data_, *weight); }

     private:
      vec_t data_;
    };
  }
}
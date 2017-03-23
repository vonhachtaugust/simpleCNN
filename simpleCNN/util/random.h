//
// Created by hacht on 3/21/17.
//

#pragma once

#include <random>

namespace simpleCNN {
  class Random_generator {
   public:
    static Random_generator& get_instance() {
      static Random_generator instance;
      return instance;
    }

    std::mt19937& operator()() { return gen_; }

    void set_seed(unsigned int seed) { gen_.seed(seed); }

   private:
    Random_generator() : gen_(1) {}
    std::mt19937 gen_;
  };

  /*template <typename T>
  inline typename std::enable_if<std::is_integral<T>::value, T>::type
  uniform_rand(T min, T max) {
    std::uniform_int_distribution<T> dst(min, max);
    return dst(Random_generator::get_instance()());
  }

  template <typename T>
  inline typename std::enable_if<std::is_floating_point<T>::value, T>::type
  uniform_rand(T min, T max) {
    std::uniform_real_distribution<T> dst(min, max);
    return dst(Random_generator::get_instance()());
  }*/

  template <typename T>
  inline T uniform_random(T min, T max) {
    std::uniform_real_distribution<T> dst(min, max);
    return dst(Random_generator::get_instance());
  }

  template <typename Iter>
  void uniform_rand(Iter begin, Iter end, float_t min, float_t max) {
    for (Iter it = begin; it != end; ++it) {
      *it = uniform_random(min, max);
    }
  }
}
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

    std::mt19937& generator() { return gen_; }
    void seed(unsigned int seed) { gen_.seed(seed); }

   private:
    Random_generator() : gen_(time(NULL)) {}
    std::mt19937 gen_;
  };

  template <typename T>
  inline T normal_dist(T mean, T std) {
    std::normal_distribution<T> dst(mean, std);
    return dst(Random_generator::get_instance().generator());
  }

  template <typename Iter>
  void normal_dist(Iter begin, Iter end, float_t mean, float_t std) {
    for (Iter it = begin; it != end; ++it) {
      *it = normal_dist(mean, std);
    }
  }

  template <typename T>
  inline T uniform_random(T min, T max) {
    std::uniform_real_distribution<T> dst(min, max);
    return dst(Random_generator::get_instance().generator());
  }

  template <typename Iter>
  void uniform_rand(Iter begin, Iter end, float_t min, float_t max) {
    for (Iter it = begin; it != end; ++it) {
      *it = uniform_random(min, max);
    }
  }

  inline bool bernoulli(float_t p) { return uniform_random(float_t{0}, float_t{1}) <= p; }

}  // namespace simpleCNN

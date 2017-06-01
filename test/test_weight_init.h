//
// Created by hacht on 4/20/17.
//

#pragma once

#include "../simpleCNN/simpleCNN.h"
#include "gtest/gtest.h"

namespace simpleCNN {

  TEST(Weight_init, random_number_generator) {
    Tensor<float_t, 1> data({10});

    auto start = data.host_begin();
    auto end   = data.host_end();

    uniform_rand(start, end, -1, 1);

    // print(data, "Data");
  }

  TEST(Weight_init, bernoulli_randomess) {
  float_t a = 0;
  float_t b = 0;
  float_t p = 0.75;

  for (size_t i = 0; i < 100000; ++i) {
    auto val = bernoulli(p);

    if (val) {
     a += 1.0f;
    } else {
      b += 1.0f;
    }
  }

  ASSERT_NEAR(a / (a+b), p, 1E-2);
  ASSERT_NEAR(b / (a+b), 1 - p, 1E-2);
  }

  TEST(Weight_init, random_between_min_max) {
  float_t min = 0;
  float_t max = 100;

  std::vector<size_t> occurrence(100);
  std::fill(occurrence.begin(), occurrence.end(), 0);

  for (size_t i = 0; i < 100000; ++i) {
    size_t val = uniform_random(min, max);
    occurrence[val]++;
  }
}

  TEST(Weight_init, Zero_mean_unit_variance) {
    /**
     * Weight initialization requires:
     *
     * E[w] = 0.
     *
     * 0.5 * n * Var[w] = 1 for all layers.
     *
     * Condition met when: Variance: sqrt(2/n).
     *
     */
    size_t num_samples = 10000;
    float_t variance   = 1.0f;
    float_t mean       = 0.0f;
    float_t step_size  = variance / (10.0f);

    std::vector<float_t> sample(num_samples);

    auto start = sample.begin();
    auto end   = sample.end();

    normal_dist(start, end, 0.0f, 1.0f);

    std::unordered_map<size_t, size_t> hist;

    size_t max_index = 0;
    for (size_t i = 0; i < num_samples; ++i) {
      float_t low  = -2;
      size_t index = 0;

      while (sample[i] > low) {
        low += step_size;
        index++;
      }
      hist[index]++;
      max_index = (index > max_index) ? index : max_index;
    }

    float_t mean_value = std::accumulate(start, end, float_t{0}) / float_t(sample.size());

    float_t variance_value = 0.0f;
    for (auto s_i : sample) {
      variance_value += (s_i - mean) * (s_i - mean);
    }
    variance_value /= float_t(sample.size());

    // print(mean_value, "Mean; ");
    // print(variance_value, "Variance; ");
    // Are you feeling lucky?
    ASSERT_NEAR(mean, mean_value, 1E-1);
    ASSERT_NEAR(variance, variance_value, 1E-1);

    /*for (size_t i = 0; i < num_samples; ++i) {
      if (i > max_index) break;
      std::cout << i << ": ";
      for (size_t j = 0; j < hist[i]; ++j) {
        std::cout << '*' << " ";
      }
      std::cout << std::endl;
    }*/
  }

}  // namespace simpleCNN

//
// Created by hacht on 5/16/17.
//

#pragma once

#include "../simpleCNN/simpleCNN.h"
#include "gtest/gtest.h"

namespace simpleCNN {

  TEST(Math, zero_mean_unit_variance) {
    tensor_t test_subject({10, 1, 10, 10});
    std::string filename = "mstd.txt";

    uniform_rand(test_subject.host_begin(), test_subject.host_end(), -1.0f, 1.0f);

    auto m = means(test_subject);
    auto s = standard_deviation(test_subject, m);

    zero_mean_unit_variance(test_subject);

    auto m_new = means(test_subject);
    auto s_new = standard_deviation(test_subject, m);

    for (auto mean : m_new) {
      ASSERT_NEAR(mean, 0, 1E0);
    }

    for (auto std : s_new) {
      ASSERT_NEAR(std, 1, 1E0);
    }

    zero_mean_unit_variance(test_subject, filename);

    auto m_load = means(test_subject);
    auto s_load = standard_deviation(test_subject, m_load);

    for (auto mean : m_load) {
      ASSERT_NEAR(mean, 0, 1E0);
    }

    for (auto std : s_load) {
      ASSERT_NEAR(std, 1, 1E0);
    }
  }

}  // namespace simpleCNN
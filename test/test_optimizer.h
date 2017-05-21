//
// Created by hacht on 4/24/17.
//

#pragma once

#include "../simpleCNN/simpleCNN.h"
#include "gtest/gtest.h"

namespace simpleCNN {

  TEST(Optimizer, Adam) {
    Adam<> opt;

    tensor_t delta({1, 1, 1, 1});

    tensor_t p({1, 1, 1, 1});
    p.fill(1.0f);

    vec_t data = {0.5, 0.3, 0.2, 0.15, 0.12, 0.09};

    int i = 0;
    float_t prev_value;
    for (auto d : data) {
      delta.host_at_index(i) = d;
      opt.update(&delta, &p, &delta, &p, 10);
      if (i > 0) {
        // print(prev_value, "Prev:" + std::to_string(i));
        // print(*p.host_begin(), "Curr: " + std::to_string(i));
        ASSERT_GE(prev_value, *p.host_begin());
      }
      prev_value = *p.host_begin();
      i++;
    }
  }
}
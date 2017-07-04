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

    // TODO: Finish test case.
  }
}  // namespace simpleCNN
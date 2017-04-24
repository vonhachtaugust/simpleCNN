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
}  // namespace simpleCNN

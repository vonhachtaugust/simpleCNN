//
// Created by hacht on 4/17/17.
//

#pragma once

#include "../simpleCNN/simpleCNN.h"
#include "gtest/gtest.h"

namespace simpleCNN {

  TEST(Activation, softmax) {
    typedef activation::Softmax<float_t> Activation;
    size_t n = 5;

    Activation h;

    tensor_t affine({1, 1, n, 1});
    vec_t affine_data = {std::log(1), std::log(2), std::log(3), std::log(4), std::log(5)};
    fill(affine_data, affine);

    tensor_t activated({1, 1, n, 1});

    h.activate(affine, activated, n);

    float_t sum = std::accumulate(activated.host_begin(), activated.host_end() + 1, float_t(0));
    ASSERT_NEAR(sum, 1, 1E-7);
  }
}
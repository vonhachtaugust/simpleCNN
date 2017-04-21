//
// Created by hacht on 4/18/17.
//

#pragma once

#include "../simpleCNN/simpleCNN.h"
#include "gtest/gtest.h"

namespace simpleCNN {

  TEST(Network, Move_semantics) {
    using Conv = Convolutional_layer<float_t, activation::ReLU<float_t>>;
    Network<Sequential> net;

    net << Conv(32, 32, 1, 1, 5, 6);
  }
}
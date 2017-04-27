//
// Created by hacht on 4/18/17.
//

#pragma once

#include "../simpleCNN/simpleCNN.h"
#include "gtest/gtest.h"

namespace simpleCNN {

  using conv = Convolutional_layer<>;
  using maxpool = Maxpooling_layer<>;
  using fully = Connected_layer<>;
  using classify = Connected_layer<>;

  TEST(Network, Move_semantics) {
    Network<Sequential> net;

    net << conv(28, 28, 1, 1, 5, 6, 1, 2, true) << maxpool(28, 28, 6, 1);
}

  TEST(Network, testing) {

}


}
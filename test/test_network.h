//
// Created by hacht on 4/18/17.
//

#pragma once

#include "../simpleCNN/simpleCNN.h"
#include "gtest/gtest.h"

namespace simpleCNN {

using dropout = Dropout_layer;
using conv    = Convolutional_layer<float_t, activation::ReLU<float_t>>;
using maxpool = Maxpooling_layer<>;
using fully   = Connected_layer<>;
using classy  = Connected_layer<float_t, activation::Softmax<float_t>>;
using network = Network<Sequential>;
using lgl     = loss::Log_likelihood<float_t>;

  TEST(Network, Move_semantics) {
    Network<Sequential> net;

    net << conv(28, 28, 1, 1, 5, 6, 1, 2, true) << maxpool(28, 28, 6, 1);
}

}
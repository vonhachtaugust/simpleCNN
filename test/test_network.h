//
// Created by hacht on 4/18/17.
//

#pragma once

#include "../simpleCNN/simpleCNN.h"
#include "gtest/gtest.h"

namespace simpleCNN {

using dropout = Dropout_layer;
using conv    = Convolutional_layer;
using maxpool = Maxpooling_layer;
using fully = Connected_layer;
using network = Network<Sequential>;
using softmax = loss::Softmax;
using relu    = activation::ReLU;

  TEST(Network, Move_semantics) {
    Network<Sequential> net;

    net << conv(28, 28, 1, 1, 5, 6, 1, 2, true) << maxpool(28, 28, 6, 1);
}

  TEST(Network, gradient_check) {
    Network<Sequential> net;
    size_t ils = 2; // input layer size
    size_t hls = 3; // hidden layer size
    size_t ols = 1; // output layer size

    size_t batch_size = 1;

    net << fully(ils, hls, 1) << relu() << fully(hls, ols, 1) << softmax();

    tensor_t input({batch_size, 1, ils, 1});
    tensor_t labels({batch_size, 1, 1, 1});

    input.host_at_index(0) = 1;
    input.host_at_index(1) = 2;

    labels.host_at_index(0) = 1;

    net.gradient_check(input, labels, batch_size);
}

}
//
// Created by hacht on 5/15/17.
//

#pragma once

#include "../simpleCNN/simpleCNN.h"
#include "gtest/gtest.h"

namespace simpleCNN {

  using fully   = Connected_layer;
  using relu    = activation::ReLU;
  using network = Network<Sequential>;
  using softmax = loss::Softmax;
  using maxpool = Maxpooling_layer;
  using dropout = Dropout_layer;
  using conv    = Convolutional_layer;

  TEST(Save, save_tensor) {
    network net1, net2;
    size_t minibatch_size = 5;

    net1 << conv(32, 32, 1, minibatch_size, 5, 6) << relu() << maxpool(28, 28, 6, minibatch_size)
         << conv(14, 14, 6, minibatch_size, 5, 16) << relu() << maxpool(10, 10, 16, minibatch_size)
         << conv(5, 5, 16, minibatch_size, 5, 120) << relu() << dropout({minibatch_size, 120, 1, 1}, 0.5)
         << fully(120, 10, minibatch_size) << softmax();

    net2 << conv(32, 32, 1, minibatch_size, 5, 6) << relu() << maxpool(28, 28, 6, minibatch_size)
         << conv(14, 14, 6, minibatch_size, 5, 16) << relu() << maxpool(10, 10, 16, minibatch_size)
         << conv(5, 5, 16, minibatch_size, 5, 120) << relu() << dropout({minibatch_size, 120, 1, 1}, 0.5)
         << fully(120, 10, minibatch_size) << softmax();

    net1.init_network();

    auto path = "test.txt";
    net1.save_to_file(path, content_type::weights_and_bias);

    net2.load_from_file(path, content_type::weights_and_bias);

    // TODO: Check both network have equal weights.
  }
}
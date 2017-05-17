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
using adam    = Adam<float_t>;

  TEST(Network, Move_semantics) {
    Network<Sequential> net;

    //net << conv(28, 28, 1, 1, 5, 6, 1, 2, true) << maxpool(28, 28, 6, 1);
}

  TEST(Network, gradient_check) {
    Network<Sequential> net;
    size_t ils = 2; // input layer size
    size_t hls = 3; // hidden layer size
    size_t ols = 3; // output layer size

    size_t batch_size = 1;

    net << fully(ils, hls, 1) << relu() << fully(hls, ols, 1) << softmax();

    tensor_t input({batch_size, 1, ils, 1});
    tensor_t labels({batch_size, 1, 1, 1});

    input.host_at_index(0) = 1;
    input.host_at_index(1) = 2;

    labels.host_at_index(0) = 0;

    auto error = net.gradient_check(input, labels);

    for (auto e : error) {
      ASSERT_NEAR(e, 1E-4, 1E-4);
    }
}

  TEST(Network, test) {
  Network<Sequential> net;
  size_t in_w = 10;
  size_t in_h = 10;
  size_t in_ch = 1;
  size_t bs = 2;
  size_t out_ch = 3;
  size_t fs = 5;

  tensor_t input({bs, in_ch, in_h, in_w});
  uniform_rand(input.host_begin(), input.host_end(), -1, 1);

  tensor_t labels({bs, 1, 1, 1});

  // w, h, in_c, batch
  net << conv(in_w, in_h, in_ch, bs, fs, out_ch) << maxpool(6, 6, out_ch, bs) << fully(3 * 3 * out_ch, out_ch, bs) << softmax();

  auto error = net.gradient_check(input, labels);

  for (auto e : error) {
    //print(e, "Error");
    ASSERT_NEAR(e, 1E-2, 1E-2);
  }
}

  TEST(Network, gradient_check_bias) {
  Network<Sequential> net;
  size_t in_w = 10;
  size_t in_h = 10;
  size_t in_ch = 1;
  size_t bs = 10;
  size_t out_ch = 3;
  size_t fs = 5;

  tensor_t input({bs, in_ch, in_h, in_w});
  uniform_rand(input.host_begin(), input.host_end(), -1, 1);

  tensor_t labels({bs, 1, 1, 1});

  // w, h, in_c, batch

  net << conv(in_w, in_h, in_ch, bs, fs, out_ch) << maxpool(6, 6, out_ch, bs) << fully(3 * 3 * out_ch, out_ch, bs) << softmax();

  auto error = net.gradient_check_bias(input, labels);
  for (auto e : error) {
    //print(e, "Error");
    ASSERT_NEAR(e, 1E-2, 1E-2);
  }

}

}
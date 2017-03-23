//
// Created by hacht on 3/16/17.
//

#pragma once

#include "../simpleCNN/simpleCNN.h"
#include "gtest/gtest.h"

namespace simpleCNN {

  inline size_t conv_out_length(size_t image_side_length, size_t filter_side_length, size_t stride, size_t padding) {
    return (image_side_length - filter_side_length + 2 * padding) / stride + 1;
  }

  TEST(Convolution, forward_propagation) {
    // http://cs231n.github.io/convolutional-networks/ - Convolution Demo /
    // Result manual verified to be correct
    size_t imageWidth   = 5;
    size_t imageHeight  = 5;
    size_t in_channels  = 3;  // 3 color channel
    size_t batch_size   = 1;
    size_t filterSize   = 3;
    size_t out_channels = 2;  // := number of filters
    size_t padding      = 1;
    size_t stride       = 2;
    bool has_bias       = true;

    Convolutional_layer<> conv(imageWidth, imageHeight, in_channels, batch_size, filterSize, out_channels, stride,
                               padding, has_bias);
    // In-data
    tensor_t image({1, in_channels, imageHeight, imageWidth});

    vec_t image_data = {0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 2, 1, 0, 1, 1, 1, 1, 2, 2, 1, 0, 2, 0, 1,
                        1, 0, 2, 2, 1, 0, 0, 1, 1, 2, 0, 0, 0, 1, 2, 2, 0, 1, 1, 0, 0, 1, 1, 0, 0,
                        2, 1, 2, 0, 1, 0, 0, 1, 2, 2, 1, 1, 1, 0, 2, 2, 0, 1, 2, 1, 2, 0, 2, 1, 2};

    vec_iter_t iter = image_data.begin();
    fill_with(iter, image);
    simple_info("Input volume:");
    std::cout << image << std::endl;

    auto assertImageIter = image_data.begin();
    for (auto d : image_data) {
      ASSERT_EQ(*assertImageIter++, d);
    }
    conv.set_in_data(image, component_t::IN_DATA);

    // Setup custom weights and bias
    vec_t weight_data = {-1, -1, 0,  1, -1, 0,  -1, -1, 0, -1, -1, -1, 0, -1, -1, 1,  0,  -1,
                         -1, 1,  0,  1, 1,  -1, 1,  -1, 0, 1,  0,  -1, 1, -1, 0,  -1, 1,  -1,
                         1,  1,  -1, 0, 1,  -1, 1,  0,  1, 0,  0,  1,  0, 1,  1,  1,  -1, 1};

    auto weightIter = weight_data.begin();
    weight_init::Test wei(weightIter, 1.0f);

    vec_t bias_data = {1, 0};

    auto biasIter = bias_data.begin();
    weight_init::Test bias(biasIter, 1.0f);

    // Weight allocation
    conv.weight_init(wei);
    conv.bias_init(bias);

    // Fill with values (depends on weight::init class)
    conv.init_weight();

    auto weights = conv.in_component(component_t::WEIGHT);
    simple_info("Weights");
    std::cout << *weights << std::endl;

    auto biases = conv.in_component(component_t::BIAS);
    simple_info("Bias");
    std::cout << *biases << std::endl;

    auto assertWeightIter = weights->host_begin();
    for (auto w : weight_data) {
      ASSERT_EQ(*assertWeightIter++, w);
    }

    auto assertBiasIter = biases->host_begin();
    for (auto b : bias_data) {
      ASSERT_EQ(*assertBiasIter++, b);
    }

    // Out-data
    tensor_t output({1, out_channels, conv_out_length(imageHeight, filterSize, stride, padding),
                     conv_out_length(imageWidth, filterSize, stride, padding)});
    conv.set_out_data(output, component_t::OUT_DATA);

    // conv.forward_propagation()
    data_ptrs_t input = {conv.in_component(component_t::IN_DATA), conv.in_component(component_t::WEIGHT),
                         conv.in_component(component_t::BIAS)};
    data_ptrs_t output_ = {conv.out_component(component_t::OUT_DATA)};
    conv.forward_propagation(input, output_);

    simple_info("Output volume");
    std::cout << output << std::endl;

    simple_error("Convolution test DONE!");
  }
}  // namespace simpleCNN

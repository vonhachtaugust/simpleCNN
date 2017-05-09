//
// Created by hacht on 4/10/17.
//

#pragma once

#include "../simpleCNN/simpleCNN.h"
#include "gtest/gtest.h"

namespace simpleCNN {

  /*
  TEST(Feedforward, forward_activation_relu) {
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

    Convolutional_layer conv(imageWidth, imageHeight, in_channels, batch_size, filterSize, out_channels, stride,
                               padding, has_bias);
    // In-data
    tensor_t image({1, in_channels, imageHeight, imageWidth});

    vec_t image_data = {0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 2, 1, 0, 1, 1, 1, 1, 2, 2, 1, 0, 2, 0, 1,
                        1, 0, 2, 2, 1, 0, 0, 1, 1, 2, 0, 0, 0, 1, 2, 2, 0, 1, 1, 0, 0, 1, 1, 0, 0,
                        2, 1, 2, 0, 1, 0, 0, 1, 2, 2, 1, 1, 1, 0, 2, 2, 0, 1, 2, 1, 2, 0, 2, 1, 2};

    fill(image_data, image);
    // simple_info("Input volume:");
    // std::cout << image << std::endl;

    conv.set_in_data(image, component_t::IN_DATA);

    // Setup custom weights and bias
    vec_t weight_data = {-1, -1, 0,  1, -1, 0,  -1, -1, 0, -1, -1, -1, 0, -1, -1, 1,  0,  -1,
                         -1, 1,  0,  1, 1,  -1, 1,  -1, 0, 1,  0,  -1, 1, -1, 0,  -1, 1,  -1,
                         1,  1,  -1, 0, 1,  -1, 1,  0,  1, 0,  0,  1,  0, 1,  1,  1,  -1, 1};
    weight_init::Test wei(weight_data);

    vec_t bias_data = {1, 0};
    weight_init::Test bias(bias_data);

    // Weight allocation
    conv.weight_init(wei);
    conv.bias_init(bias);

    // Fill with values (depends on weight::init class)
    conv.init_weight();

    // Out-data
    tensor_t output({1, out_channels, conv_out_length(imageHeight, filterSize, stride, padding),
                     conv_out_length(imageWidth, filterSize, stride, padding)});
    conv.set_out_data(output, component_t::OUT_DATA);
    conv.set_out_data(output, component_t::AUX);

    // conv.forward_propagation()
    data_ptrs_t input = {conv.in_component(component_t::IN_DATA), conv.in_component(component_t::WEIGHT),
                         conv.in_component(component_t::BIAS)};
    data_ptrs_t output_ = {conv.out_component(component_t::OUT_DATA), conv.out_component(component_t::AUX)};
    conv.forward_propagation(input, output_);

    // print(*output_[1], "Non-activated output");
    //conv.forward_activation(*output_[0], *output_[1]);
    // print(*output_[1], "Activated output");
  }

  TEST(Feedforward, backward_activation) {
    size_t in_dim     = 4;
    size_t out_dim    = 2;
    size_t batch_size = 1;

    Connected_layer con(in_dim, out_dim, batch_size);

    // Input
    tensor_t in({batch_size, 1, in_dim, 1});
    vec_t in_data = {1, 2, -2, -1};
    fill(in_data, in);

    // Weight, bias
    vec_t weight_data = {1, 1, 1, 1, -1, -1, -1, -1};
    vec_t bias_data   = {1, -1};

    weight_init::Test weight(weight_data);
    weight_init::Test bias(bias_data);

    con.weight_init(weight);
    con.bias_init(bias);
    con.init_weight();

    tensor_t dW({batch_size, 1, in_dim, out_dim});
    tensor_t db({batch_size, 1, out_dim, 1});

    // curr grad
    tensor_t curr_grad({batch_size, 1, out_dim, 1});
    vec_t curr_grad_data = {1, -1};
    fill(curr_grad_data, curr_grad);

    // prev grad
    tensor_t prev_grad({batch_size, 1, in_dim, 1});

    data_ptrs_t input    = {&in, con.in_component(component_t::WEIGHT), con.in_component(component_t::BIAS)};
    data_ptrs_t output   = {};
    data_ptrs_t in_grad  = {&prev_grad, &dW, &db};
    data_ptrs_t out_grad = {&curr_grad, &curr_grad};

    con.back_propagation(input, output, in_grad, out_grad);

    // In should be the affine transformation, not the activated values. Here it is
    // assumed that in are the z values and not the activate(z) values.
    // The prev grad are the values that are activated.
    // print(*in_grad[0], "Non-activated");
    //con.backward_activation(*input[0], *in_grad[0], *in_grad[0]);
    // print(*in_grad[0], "Activated");

    vec_t correct = {2, 2, 0, 0};
    auto iter     = in_grad[0]->host_begin();
    for (const auto& d : correct) {
      ASSERT_EQ(*iter++, d);
    }
  } */
}
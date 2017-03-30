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

  TEST(Convolution, flip_filters) {
    tensor_t weights({2, 3, 3, 3});

    vec_t weight_data = {-1, -1, 0,  1, -1, 0,  -1, -1, 0, -1, -1, -1, 0, -1, -1, 1,  0,  -1,
                         -1, 1,  0,  1, 1,  -1, 1,  -1, 0, 1,  0,  -1, 1, -1, 0,  -1, 1,  -1,
                         1,  1,  -1, 0, 1,  -1, 1,  0,  1, 0,  0,  1,  0, 1,  1,  1,  -1, 1};

    //vec_iter_t iter = weight_data.begin();
    fill_with(weight_data, weights);
    // simple_info("Before flip: ");
    // std::cout << weights << std::endl;

    matrix_t mRows(
      {2, weights.dimension(dim_t::depth) * weights.dimension(dim_t::height) * weights.dimension(dim_t::width)});
    im2row_flipped_cpu(weights, mRows, weights.dimension(dim_t::stack), weights.dimension(dim_t::depth),
                       weights.dimension(dim_t::height), weights.dimension(dim_t::width));
    //auto mIter = mRows.host_begin();
    //fill_with(mRows, weights);
    // simple_info("After flip: ");
    // std::cout << weights << std::endl;

    // TODO: Implement flipped assertion test
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

    fill_with(image_data, image);
    // simple_info("Input volume:");
    // std::cout << image << std::endl;

    auto assertImageIter = image_data.begin();
    for (auto d : image_data) {
      ASSERT_EQ(*assertImageIter++, d);
    }
    conv.set_in_data(image, component_t::IN_DATA);

    // Setup custom weights and bias
    vec_t weight_data = {-1, -1, 0,  1, -1, 0,  -1, -1, 0, -1, -1, -1, 0, -1, -1, 1,  0,  -1,
                         -1, 1,  0,  1, 1,  -1, 1,  -1, 0, 1,  0,  -1, 1, -1, 0,  -1, 1,  -1,
                         1,  1,  -1, 0, 1,  -1, 1,  0,  1, 0,  0,  1,  0, 1,  1,  1,  -1, 1};
    weight_init::Test wei(weight_data, 1.0f);

    vec_t bias_data = {1, 0};
    weight_init::Test bias(bias_data, 1.0f);

    // Weight allocation
    conv.weight_init(wei);
    conv.bias_init(bias);

    // Fill with values (depends on weight::init class)
    conv.init_weight();

    auto weights = conv.in_component(component_t::WEIGHT);
    // simple_info("Weights");
    // std::cout << *weights << std::endl;

    auto biases = conv.in_component(component_t::BIAS);
    // simple_info("Bias");
    // std::cout << *biases << std::endl;

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

    // simple_info("Output volume");
    // std::cout << output << std::endl;
  }

  TEST(Convolution, backpropagation_deltas) {
  size_t imageWidth   = 3;
  size_t imageHeight  = 3;
  size_t in_channels  = 3;  // 3 color channel
  size_t batch_size   = 1;
  size_t filterSize   = 3;
  size_t out_channels = 2;  // := number of filters
  size_t padding      = filterSize - 1;
  size_t stride       = 1;
  bool has_bias       = true;

  tensor_t delta({1, 2, 3, 3});
  vec_t ddata = {-3, 1, 1, 0, -7, 1, -6, -1, 3, 1, 3 ,5, 0, 4, -1, -3, 5 ,2};
  fill_with(ddata, delta);

  // 9 index from first channel, 9 index from second channel => 18 rows total.
  // 25 locations for each index to participate => 25 rows.
  matrix_t delta_as_matrix({18, 25});

  delta2matrix_cpu(delta, 0, delta_as_matrix, out_channels, imageHeight, imageWidth, filterSize, stride, padding);
  //simple_info("Deltas as matrix: ");
  //std::cout << delta_as_matrix << std::endl;

  tensor_t weight({2, 3, 3, 3});
  vec_t weight_data = {-1, -1, 0,  1, -1, 0,  -1, -1, 0, -1, -1, -1, 0, -1, -1, 1,  0,  -1,
                       -1, 1,  0,  1, 1,  -1, 1,  -1, 0, 1,  0,  -1, 1, -1, 0,  -1, 1,  -1,
                       1,  1,  -1, 0, 1,  -1, 1,  0,  1, 0,  0,  1,  0, 1,  1,  1,  -1, 1};
  fill_with(weight_data, weight);

  matrix_t weight_as_matrix({3, 18});

  weight2matrix_cpu(weight, weight_as_matrix, out_channels, in_channels, filterSize, filterSize, filterSize);
  //simple_info("Weights as matrix: ");
  //std::cout << weight_as_matrix << std::endl;

  matrix_t result({3, 25});
  multiply_2_dim_tensors_float(weight_as_matrix, delta_as_matrix, result, false, false);
  //simple_info("Matrix multiplication result: ");
  //std::cout << result << std::endl;

  tensor_t tensor({1, 3, 5, 5});
  col2im_cpu(result, 0, tensor, tensor.dimension(dim_t::depth), tensor.dimension(dim_t::height), tensor.dimension(dim_t::width));
  simple_info("Result as tensor: ");
  std::cout << tensor << std::endl;
  }
}  // namespace simpleCNN

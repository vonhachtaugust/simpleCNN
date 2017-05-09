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

  /*
  TEST(Convolution, flip_filters) {
  // TODO: Flip implicitly made during tensor to matrix conversion. This tests is deprecated ...
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

    // TODO: Deprecated function call ...
    im2row_flipped_cpu(weights, mRows, weights.dimension(dim_t::stack), weights.dimension(dim_t::depth),
                       weights.dimension(dim_t::height), weights.dimension(dim_t::width));
    //auto mIter = mRows.host_begin();
    //fill(mRows, weights);
    // simple_info("After flip: ");
    // std::cout << weights << std::endl;

    // TODO: Implement flipped assertion test
  }*/

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

    auto assertImageIter = image_data.begin();
    for (const auto& d : image_data) {
      ASSERT_EQ(*assertImageIter++, d);
    }
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

    auto weights = conv.in_component_data(component_t::WEIGHT);
    // simple_info("Weights");
    // std::cout << *weights << std::endl;

    auto biases = conv.in_component_data(component_t::BIAS);
    // simple_info("Bias");
    // std::cout << *biases << std::endl;

    auto assertWeightIter = weights->host_begin();
    for (const auto& w : weight_data) {
      ASSERT_EQ(*assertWeightIter++, w);
    }

    auto assertBiasIter = biases->host_begin();
    for (const auto& b : bias_data) {
      ASSERT_EQ(*assertBiasIter++, b);
    }

    // Out-data
    tensor_t output({1, out_channels, conv_out_length(imageHeight, filterSize, stride, padding),
                     conv_out_length(imageWidth, filterSize, stride, padding)});
    conv.set_out_data(output, component_t::OUT_DATA);

    // conv.forward_propagation()
    data_ptrs_t input = {conv.in_component_data(component_t::IN_DATA), conv.in_component_data(component_t::WEIGHT),
                         conv.in_component_data(component_t::BIAS)};
    data_ptrs_t output_ = {conv.out_component_data(component_t::OUT_DATA)};
    conv.forward_propagation(input, output_);
    // simple_info("Output volume");
    // std::cout << output << std::endl;

    vec_t correct_output         = {0, -4, 1, -3, -3, -6, 0, -4, -3, 4, 1, 2, -1, 3, 9, 1, 3, 4};
    auto assertCorrectOutputIter = output.host_begin();
    for (const auto& co : correct_output) {
      ASSERT_EQ(*assertCorrectOutputIter++, co);
    }
  }

  TEST(Convolution, backpropagation_deltas_raw) {
    size_t imageWidth   = 5;
    size_t imageHeight  = 5;
    size_t in_channels  = 3;  // 3 color channel
    size_t batch_size   = 1;
    size_t filterSize   = 3;
    size_t out_channels = 2;  // := number of filters
    size_t padding      = filterSize - 1;
    size_t stride       = 1;
    bool has_bias       = true;

    tensor_t delta({1, 2, 3, 3});
    vec_t ddata = {-3, 1, 1, 0, -7, 1, -6, -1, 3, 1, 3, 5, 0, 4, -1, -3, 5, 2};
    fill(ddata, delta);

    int outputWidth  = 3;
    int outputHeight = 3;

    // 9 index from first channel, 9 index from second channel => 18 rows total.
    // 25 locations for each index to participate => 25 rows.
    matrix_t delta_as_matrix({18, 25});

    im2col_cpu(delta, 0, delta_as_matrix, out_channels, outputHeight, outputWidth, filterSize, stride, padding);
    // simple_info("Deltas as matrix: ");
    // std::cout << delta_as_matrix << std::endl;

    tensor_t weight({2, 3, 3, 3});
    vec_t weight_data = {-1, -1, 0,  1, -1, 0,  -1, -1, 0, -1, -1, -1, 0, -1, -1, 1,  0,  -1,
                         -1, 1,  0,  1, 1,  -1, 1,  -1, 0, 1,  0,  -1, 1, -1, 0,  -1, 1,  -1,
                         1,  1,  -1, 0, 1,  -1, 1,  0,  1, 0,  0,  1,  0, 1,  1,  1,  -1, 1};
    fill(weight_data, weight);

    matrix_t weight_as_matrix({3, 18});

    im2row_flipped_cpu(weight, weight_as_matrix, out_channels, in_channels, filterSize, filterSize);
    // simple_info("Weights as matrix: ");
    // std::cout << weight_as_matrix << std::endl;

    matrix_t result({3, 25});
    sgemm(weight_as_matrix, delta_as_matrix, result, false, false);
    // simple_info("Matrix multiplication result: ");
    // std::cout << result << std::endl;

    tensor_t tensor({1, 3, 5, 5});
    col2im_insert_cpu(result, 0, tensor, tensor.dimension(dim_t::depth), tensor.dimension(dim_t::height),
               tensor.dimension(dim_t::width));
    // simple_info("Result as tensor: ");
    // std::cout << tensor << std::endl;
  }

  TEST(Convolution, backpropagation_weight_raw) {
    // NOTE: Raw version
    size_t imageWidth   = 5;
    size_t imageHeight  = 5;
    size_t outWidth     = 3;
    size_t outHeight    = 3;
    size_t in_channels  = 3;  // 3 color channel
    size_t batch_size   = 1;
    size_t filterSize   = 3;
    size_t out_channels = 2;  // := number of filters
    size_t padding      = 1;
    size_t stride       = 2;
    bool has_bias       = true;

    tensor_t delta({1, 2, 3, 3});
    vec_t ddata = {-3, 1, 1, 0, -7, 1, -6, -1, 3, 1, 3, 5, 0, 4, -1, -3, 5, 2};
    fill(ddata, delta);

    tensor_t image({1, 3, 5, 5});
    vec_t idata = {0, 1, 1, 2, 1, 1, 1, 2, 2, 0, 2, 0, 1, 1, 1, 2, 1, 1, 2, 0, 0, 1, 2, 2, 2,
                   2, 0, 2, 0, 1, 0, 0, 0, 2, 1, 1, 1, 0, 0, 2, 0, 0, 0, 0, 1, 1, 2, 0, 2, 2,
                   1, 1, 1, 1, 1, 1, 0, 2, 0, 2, 0, 1, 2, 0, 0, 1, 2, 0, 1, 0, 0, 1, 1, 1, 1};
    fill(idata, image);

    matrix_t mImage({in_channels * filterSize * filterSize, outWidth * outHeight});
    im2col_cpu(image, 0, mImage, in_channels, imageHeight, imageWidth, filterSize, stride, padding);
    // std::cout << mImage << std::endl;

    matrix_t mDelta({out_channels, outWidth * outHeight});
    im2col_cpu(delta, 0, mDelta, out_channels, outHeight, outWidth);
    // std::cout << mDelta << std::endl;

    matrix_t mResult({mImage.shape()[0], mDelta.shape()[0]});
    sgemm(mImage, mDelta, mResult, false, true);
    // std::cout << mResult << std::endl;

    tensor_t result({out_channels, in_channels, filterSize, filterSize});
    row2im_add_cpu(mResult, result, out_channels, in_channels, filterSize, filterSize);
    // std::cout << result << std::endl;

    /*for (size_t i = 0; i < result.dimension(dim_t::batch); ++i)
    {
      for (size_t j = 0; j < result.dimension(dim_t::depth); ++j)
      {
        auto start = result.host_iter(i,j,0,0);
        auto end = start + result.dimension(dim_t::height) * result.dimension(dim_t::width);
        for (; start != end; ++start)
        {
          std::cout << *start << "\t";
        }
        std::cout << std::endl;
      }
    }*/
  }

  TEST(Convolution, backprop_op) {
    size_t imageWidth   = 5;
    size_t imageHeight  = 5;
    size_t outWidth     = 3;
    size_t outHeight    = 3;
    size_t in_channels  = 3;  // 3 color channel
    size_t batch_size   = 1;
    size_t filterSize   = 3;
    size_t out_channels = 2;  // := number of filters
    size_t padding      = 1;
    size_t stride       = 2;
    bool has_bias       = true;

    Convolutional_layer conv(imageWidth, imageHeight, in_channels, batch_size, filterSize, out_channels, stride,
                               padding, has_bias);

    tensor_t input_previous_layer({1, in_channels, imageHeight, imageWidth});
    vec_t input_data = {0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 2, 1, 0, 1, 1, 1, 1, 2, 2, 1, 0, 2, 0, 1,
                        1, 0, 2, 2, 1, 0, 0, 1, 1, 2, 0, 0, 0, 1, 2, 2, 0, 1, 1, 0, 0, 1, 1, 0, 0,
                        2, 1, 2, 0, 1, 0, 0, 1, 2, 2, 1, 1, 1, 0, 2, 2, 0, 1, 2, 1, 2, 0, 2, 1, 2};
    fill(input_data, input_previous_layer);

    tensor_t curr_delta({1, 2, 3, 3});
    vec_t curr_delta_data = {0, -4, 1, -3, -3, -6, 0, -4, -3, 4, 1, 2, -1, 3, 9, 1, 3, 4};
    fill(curr_delta_data, curr_delta);

    // Setup custom weights and bias
    vec_t weight_data = {-1, -1, 0,  1, -1, 0,  -1, -1, 0, -1, -1, -1, 0, -1, -1, 1,  0,  -1,
                         -1, 1,  0,  1, 1,  -1, 1,  -1, 0, 1,  0,  -1, 1, -1, 0,  -1, 1,  -1,
                         1,  1,  -1, 0, 1,  -1, 1,  0,  1, 0,  0,  1,  0, 1,  1,  1,  -1, 1};
    weight_init::Test wei(weight_data);

    vec_t bias_data = {1, 0};
    weight_init::Test bias(bias_data);

    // Weight allocation (necessary for custom weights)
    conv.weight_init(wei);
    conv.bias_init(bias);

    // Fill with values (depends on weight::init class)
    conv.init_weight();

    tensor_t prev_delta({1, in_channels, imageHeight, imageWidth});
    tensor_t dW({out_channels, in_channels, filterSize, filterSize});
    tensor_t dB({out_channels, 1, 1, 1});

    data_ptrs_t input = {&input_previous_layer, conv.in_component_data(component_t::WEIGHT),
                         conv.in_component_data(component_t::BIAS)};
    data_ptrs_t output    = {};
    data_ptrs_t in_grads  = {&prev_delta, &dW, &dB};
    data_ptrs_t out_grads = {&curr_delta, &curr_delta};

    conv.back_propagation(input, output, in_grads, out_grads);

    /*
    simple_info("input gradients: ");
    std::cout << *in_grads[0] << std::endl;

    simple_info("input dW: ");
    std::cout << *in_grads[1] << std::endl;

    simple_info("input dB: ");
    std::cout << *in_grads[2] << std::endl;

    simple_info("output gradients: ");
    std::cout << *out_grads[1] << std::endl;
     */
  }

  TEST(Convolution, backprop_op_II) {
    size_t imageWidth   = 5;
    size_t imageHeight  = 5;
    size_t outWidth     = 3;
    size_t outHeight    = 3;
    size_t in_channels  = 1;  // 1 color channel
    size_t batch_size   = 2;
    size_t filterSize   = 3;
    size_t out_channels = 1;  // := number of filters
    size_t padding      = 1;
    size_t stride       = 2;
    bool has_bias       = true;

    Convolutional_layer conv(imageWidth, imageHeight, in_channels, batch_size, filterSize, out_channels, stride,
                               padding, has_bias);

    tensor_t input_previous_layer({batch_size, in_channels, imageHeight, imageWidth});
    vec_t input_data = {0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 2, 1, 0, 1, 1, 1, 1, 2, 2, 1, 0, 2, 0, 1,
                        1, 0, 2, 2, 1, 0, 0, 1, 1, 2, 0, 0, 0, 1, 2, 2, 0, 1, 1, 0, 0, 1, 1, 0, 0};
    fill(input_data, input_previous_layer);

    tensor_t curr_delta({batch_size, out_channels, outHeight, outWidth});
    vec_t curr_delta_data = {0, -4, 1, -3, -3, -6, 0, -4, -3, 4, 1, 2, -1, 3, 9, 1, 3, 4};
    fill(curr_delta_data, curr_delta);

    // Setup custom weights and bias [weight = 1 x 1 x 3 x 3]
    vec_t weight_data = {-1, -1, 0, 1, -1, 0, -1, -1, 0, -1, -1, -1, 0, -1, -1, 1, 0, -1};
    weight_init::Test wei(weight_data);

    vec_t bias_data = {1};
    weight_init::Test bias(bias_data);

    // Weight allocation (necessary for custom weights)
    conv.weight_init(wei);
    conv.bias_init(bias);

    // Fill with values (depends on weight::init class)
    conv.init_weight();

    tensor_t prev_delta({batch_size, in_channels, imageHeight, imageWidth});
    tensor_t dW({out_channels, in_channels, filterSize, filterSize});
    tensor_t dB({out_channels, 1, 1, 1});

    data_ptrs_t input = {&input_previous_layer, conv.in_component_data(component_t::WEIGHT),
                         conv.in_component_data(component_t::BIAS)};
    data_ptrs_t output    = {};
    data_ptrs_t in_grads  = {&prev_delta, &dW, &dB};
    data_ptrs_t out_grads = {&curr_delta, &curr_delta};

    conv.back_propagation(input, output, in_grads, out_grads);

    /* Tensor(0,0,:,:): / Non-add
      -19	-13	-14
        -6	-24	-6
        -18	-18	-13
      Tensor(1,0,:,:):
      13	26	6
      16	29	6
      11	6	4 */

    /*
    Tensor(0,0,:,:): / Add
      -19	-13	-14
       -6	-24	-6
       18 -18	-13
    Tensor(1,0,:,:):
      0	0	0
      0	0	0
      0	0	0

    Tensor(0,0,:,:):
      -6	13	-8
      10	5	0
      -7	-12	-9
    Tensor(1,0,:,:):
      0	0	0
      0	0	0
      0	0	0 */

    /*
      simple_info("input gradients: ");
      std::cout << *in_grads[0] << std::endl;

      simple_info("input dW: ");
      std::cout << *in_grads[1] << std::endl;

      simple_info("input dB: ");
      std::cout << *in_grads[2] << std::endl;

      simple_info("output gradients: ");
      std::cout << *out_grads[1] << std::endl;*/
  }

  TEST(Convolution, backprop_op_III) {
  using conv = Convolutional_layer;
  using classy = Connected_layer;

  size_t in_w = 5;
  size_t in_h = 5;
  size_t in_ch = 16;
  size_t b = 1;
  size_t f = 5;
  size_t out_ch = 120;

  tensor_t input({b, in_ch, in_h, in_w});
  tensor_t W({out_ch, in_ch, in_h, in_w});
  tensor_t B({out_ch, 1, 1, 1});

  tensor_t dW({out_ch, in_ch, in_h, in_w});
  tensor_t db({out_ch, 1, 1, 1});

  tensor_t output({b, out_ch, 1, 1});
  tensor_t output_a({b, out_ch, 1, 1});

  tensor_t in_grad({b, in_ch, in_h, in_w});
  tensor_t out_grad({b, out_ch, 1, 1});
  tensor_t out_grad_a({b, out_ch, 1, 1});

  conv c(in_w, in_h, in_ch, b, f, out_ch);

  data_ptrs_t in_d = {&input, &W, &B};
  data_ptrs_t out_d = {&output, &output_a};
  data_ptrs_t in_g = {&in_grad, &dW, &db};
  data_ptrs_t out_g = {&out_grad, &out_grad_a};

  c.back_propagation(in_d, out_d, in_g, out_g);
}

}  // namespace simpleCNN

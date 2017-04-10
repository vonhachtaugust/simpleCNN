//
// Created by hacht on 4/4/17.
//

#pragma once

#include "../simpleCNN/simpleCNN.h"
#include "gtest/gtest.h"

namespace simpleCNN {

  TEST(Maxpooling, forward_propagation) {
    size_t in_width       = 4;
    size_t in_height      = 4;
    size_t in_channels    = 2;
    size_t batch_size     = 2;
    size_t pooling_size_x = 2;
    size_t pooling_size_y = 2;

    Maxpooling_layer<> maxpool(in_width, in_height, in_channels, batch_size);

    vec_t in_data = {1, 1, 2, 4, 5, 6, 7, 8, 3, 2, 1, 0, 1, 2, 3, 4,
                     1, 1, 2, 4, 5, 6, 7, 8, 3, 2, 1, 0, 1, 2, 3, 4,
                     1, 1, 2, 4, 5, 6, 7, 8, 3, 2, 1, 0, 1, 2, 3, 4,
                     1, 1, 2, 4, 5, 6, 7, 8, 3, 2, 1, 0, 1, 2, 3, 4};
    tensor_t img({batch_size, in_channels, in_height, in_width});
  fill(in_data, img);
    //simple_info("In data: ");
    //std::cout << img << std::endl;

    tensor_t out({batch_size, in_channels, pooling_size_y, pooling_size_x});
    //maxpool.set_in_data(img, component_t::IN_DATA);
    //maxpool.set_out_data(out, component_t::OUT_DATA);
    //maxpool.set_out_data(out, component_t::AUX);

    //data_ptrs_t input  = {maxpool.in_component(component_t::IN_DATA)};
    //data_ptrs_t output = {maxpool.out_component(component_t::OUT_DATA),
    //                      maxpool.out_component(component_t::AUX)};

    data_ptrs_t input = {&img};
  data_ptrs_t output = {&out, &out};
    maxpool.forward_propagation(input, output);
    //simple_info("Out data: ");
    //std::cout << out << std::endl;

    auto outIter        = out.host_begin();
    vec_t correctOutput = {6, 8, 3, 4, 6, 8, 3, 4, 6, 8, 3, 4, 6, 8, 3, 4};
    for (const auto& d : correctOutput) {
      ASSERT_EQ(*outIter++, d);
    }
  }

  TEST(Maxpooling, back_propagation) {
  size_t in_width       = 4;
  size_t in_height      = 4;
  size_t in_channels    = 1;
  size_t out_channels = in_channels; // for clarity
  size_t batch_size     = 1;
  size_t pooling_size_x = 2;
  size_t pooling_size_y = 2;

  Maxpooling_layer<> maxpool(in_width, in_height, in_channels, batch_size);

  // In order to get max index tensor we have to perform a forward pass first.
  vec_t in_data = {1, 1, 2, 4, 5, -1, 7, -1, 3, 5, 1, 0, 1, 2, 3, 4};
  tensor_t img({batch_size, in_channels, in_height, in_width});
  fill(in_data, img);
  //simple_info("In data: ");
  //std::cout << img << std::endl;

  tensor_t out({batch_size, in_channels, pooling_size_y, pooling_size_x});

  data_ptrs_t input  = {&img};
  data_ptrs_t output = {&out, &out};
  maxpool.forward_propagation(input, output);
  //simple_info("Out data: ");
  //std::cout << out << std::endl;

  tensor_t curr_delta({batch_size, out_channels, pooling_size_y, pooling_size_x});
  vec_t curr_delta_data = {1, 2, 3, 4};
  fill(curr_delta_data, curr_delta);
  //simple_info("Current delta: ");
  //std::cout << curr_delta << std::endl;

  tensor_t prev_delta({batch_size, in_channels, in_height, in_width});
  data_ptrs_t input_grad = {&curr_delta};
  data_ptrs_t output_grad = {&prev_delta, &prev_delta};
  maxpool.back_propagation(input, output, input_grad, output_grad);
  //simple_info("Prev delta:");
  //std::cout << prev_delta << std::endl;

  ASSERT_EQ(prev_delta.host_at(0,0,1, 0), curr_delta_data[0]);
  ASSERT_EQ(prev_delta.host_at(0,0,1,2), curr_delta_data[1]);
  ASSERT_EQ(prev_delta.host_at(0,0,2,1), curr_delta_data[2]);
  ASSERT_EQ(prev_delta.host_at(0,0,3,3), curr_delta_data[3]);
}

}

//
// Created by hacht on 4/27/17.
//

#pragma once

#include "../simpleCNN/simpleCNN.h"
#include "gtest/gtest.h"

namespace simpleCNN {

  TEST(Dropout, forward_propagation) {
    size_t batch_size = 2;
    size_t col        = 3;

    Dropout_layer dNet(0.5);
    dNet.set_in_shape({batch_size, 1, col, 1});

    tensor_t input({batch_size, 1, col, 1});
    tensor_t output({batch_size, 1, col, 1});

    for (size_t i = 0; i < batch_size; ++i) {
      for (size_t j = 0; j < col; ++j) {
        input.host_at_index(i * col + j) = j + 1;
      }
    }
    // print(input, "Input");

    data_ptrs_t name1 = {&input};
    data_ptrs_t name2 = {&output};

    dNet.forward_propagation(name1, name2);
    // print(output, "Output after forwardprop");
  }

  TEST(Dropout, backward_propagation) {
    size_t batch_size = 2;
    size_t col        = 3;

    Dropout_layer dNet(0.5);
    dNet.set_in_shape({batch_size, 1, col, 1});

    tensor_t input({batch_size, 1, col, 1});
    tensor_t output({batch_size, 1, col, 1});

    for (size_t i = 0; i < batch_size; ++i) {
      for (size_t j = 0; j < col; ++j) {
        input.host_at_index(i * col + j) = j + 1;
      }
    }
    // print(input, "Input");

    data_ptrs_t name1 = {&input};
    data_ptrs_t name2 = {&output};

    dNet.forward_propagation(name1, name2);
    // print(output, "Output after forwardprop");

    data_ptrs_t curr_delta = {&output};

    tensor_t prev_delta_t({batch_size, 1, col, 1});
    data_ptrs_t prev_delta = {&prev_delta_t};

    dNet.back_propagation(name1, name2, prev_delta, curr_delta);
    // print(prev_delta_t, "Previous delta");
  }
}  //  namespace simpleCNN

//
// Created by hacht on 4/6/17.
//

#pragma once

#include "../simpleCNN/simpleCNN.h"
#include "gtest/gtest.h"

namespace simpleCNN {

  TEST(Tensor, indexing) {
    size_t w = 3;
    size_t h = 3;
    size_t c = 1;
    size_t b = 3;

    tensor_t tensor({b, c, h, w});

    vec_t data(tensor.size());
    for (int i = 0; i < b * c * h * w; ++i) {
      data[i] = i;
    }
    fill(data, tensor);

    ASSERT_EQ(tensor.host_at(1, 0, 2, 1), tensor.host_index(16));
  }

  TEST(Tensor, resize) {
    size_t w = 3;
    size_t h = 3;
    size_t c = 3;

    vec_t data;
    for (int i = 0; i < w * h * c; ++i) data.push_back(i);

    tensor_t tensor({1, c, h, w});
    fill(data, tensor);
    // print(tensor, "Original");

    tensor.resize({1, 1, 1, c * h * w});
    // print(tensor, "Resized");

    auto iter = tensor.host_begin();
    for (const auto& d : data) {
      ASSERT_EQ(*iter++, d);
    }
  }

  TEST(Tensor, reshape) {
    size_t b = 2;
    size_t c = 2;
    size_t h = 2;
    size_t w = 2;

    vec_t data;
    for (int i = 0; i < b * c * h * w; ++i) data.push_back(i);

    tensor_t tensor({b, c, h, w});
    fill(data, tensor);
    // print(tensor, "Original");

    std::array<size_t, 4> array = {b, 1, 1, c * h * w};
    std::vector<size_t> vector({b, c * h * w, 1, 1});

    tensor.reshape(array);
    // print(tensor, "Reshaped using array");

    tensor.reshape(vector);
    // print(tensor, "Reshaped using vector");
    auto iter = tensor.host_begin();
    for (const auto& d : data) {
      ASSERT_EQ(*iter++, d);
    }
  }
}

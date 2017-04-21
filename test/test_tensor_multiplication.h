//
// Created by hacht on 4/13/17.
//

#pragma once

#include "../simpleCNN/simpleCNN.h"
#include "gtest/gtest.h"

namespace simpleCNN {

  TEST(Dot_product, tensor_dot_product_I) {
    size_t batch   = 1;
    size_t channel = 1;
    size_t height  = 3;
    size_t width   = 3;

    tensor_t A({batch, channel, height, width});
    vec_t A_data = {0, 2, 1, 0, 4, 1, 0, 0, 1};
    fill(A_data, A);

    tensor_t B({1, channel, height, width});
    vec_t B_data = {1, 0, 1, 0, -1, 0, -1, 0, 1};
    fill(B_data, B);

    float AB;
    for (size_t i = 0; i < batch; ++i) {
      for (size_t j = 0; j < channel; ++j) {
        auto start_A = A.host_ptr(i, j, 0, 0);
        auto start_B = B.host_ptr(0, j, 0, 0);

        AB = dot(A, start_A, B, start_B);
      }
    }
    // print(AB, "AB");
    ASSERT_EQ(AB, -2);
  }

  TEST(Multiplication, tensor_multiplication_I) {
    size_t batch   = 1;
    size_t channel = 1;
    size_t height  = 3;
    size_t width   = 3;

    tensor_t A({batch, channel, height, width});
    vec_t A_data = {0, 2, 1, 0, 2, 1, 0, 0, 1};
    fill(A_data, A);

    tensor_t B({1, channel, height, width});
    vec_t B_data = {1, 1, 1, 1, 1, 1, 1, 1, 1};
    fill(B_data, B);

    tensor_t AB({batch, channel, height, width});
    for (size_t i = 0; i < batch; ++i) {
      for (size_t j = 0; j < channel; ++j) {
        auto start_A  = A.host_ptr(i, j, 0, 0);
        auto start_B  = B.host_ptr(0, j, 0, 0);
        auto start_AB = AB.host_ptr(i, j, 0, 0);

        multiply(A, start_A, B, start_B, start_AB, false, false);
      }
    }
    // print(AB, "AB");

    vec_t correct = {3, 3, 3, 3, 3, 3, 1, 1, 1};
    auto iter     = AB.host_begin();
    for (const auto& d : correct) {
      ASSERT_EQ(*iter++, d);
    }
  }

  TEST(Multiplication, tensor_multiplication_II) {
    size_t batch   = 2;  // three images
    size_t channel = 3;  // rgb
    size_t height  = 3;
    size_t width   = 3;

    tensor_t A({batch, channel, height, width});
    vec_t A_data = {0, 2, 1, 0, 2, 1, 0, 0, 1, 2, 0, 0, 1, 1, 2, 1, 1, 0, 1, 2, 2, 0, 2, 1, 1, 1, 2,
                    1, 1, 0, 2, 2, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 2, 2, 0, 2, 1, 0, 0, 0, 0};
    fill(A_data, A);

    tensor_t B({1, channel, height, width});
    vec_t B_data = {1, 0, 1, 0, -1, 0, -1, 0, 1, 0, 1, 0, -1, -1, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0, -1, 1};
    fill(B_data, B);
    tensor_t AB({batch, channel, height, width});

    for (size_t i = 0; i < batch; ++i) {
      for (size_t j = 0; j < channel; ++j) {
        auto start_A  = A.host_ptr(i, j, 0, 0);
        auto start_B  = B.host_ptr(0, j, 0, 0);
        auto start_AB = AB.host_ptr(i, j, 0, 0);

        multiply(A, start_A, B, start_B, start_AB, false, false);
      }
    }
    // print(AB, "AB");
  }

}  // namespace simpleCNN

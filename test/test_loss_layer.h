//
// Created by hacht on 4/13/17.
//

#pragma once

#include "../simpleCNN/simpleCNN.h"
#include "gtest/gtest.h"

namespace simpleCNN {
  TEST(Loss, regularization) {
    tensor_t weight({1, 1, 3, 2});
    vec_t data = {1, 2, 3, 1, 2, 3};
    fill(data, weight);

    std::vector<tensor_t *> weights = {&weight};

    float_t result = regularization<float_t>(weights);
    ASSERT_EQ(result, 28);
  }

  TEST(Loss, loss_function) {
    size_t n      = 3;
    shape4d shape = {1, 1, n, 1};
    loss::Softmax ll;
    ll.set_shape(shape);

    tensor_t in_data({1, 1, n, 1});
    vec_t test_data = {std::log(1), std::log(2), std::log(3)};
    fill(test_data, in_data);

    tensor_t out_data({1, 1, n, 1});

    ll.loss_function(in_data, out_data);

    ASSERT_NEAR(out_data.host_at_index(0), float_t(1) / float_t(6), 1E-6);
    ASSERT_NEAR(out_data.host_at_index(1), float_t(2) / float_t(6), 1E-6);
    ASSERT_NEAR(out_data.host_at_index(2), float_t(3) / float_t(6), 1E-6);
    ASSERT_NEAR(std::accumulate(out_data.host_begin(), out_data.host_end(), float_t(0)), 1, 1E-6);
  }

  TEST(Loss, loss_gradient) {
    size_t n      = 3;
    shape4d shape = {1, 1, n, 1};
    loss::Softmax ll;
    ll.set_shape(shape);

    tensor_t output({1, 1, n, 1});
    vec_t test_data = {0.2, 0.3, 0.5};
    fill(test_data, output);

    tensor_t target({1, 1, 1, 1});
    vec_t target_data = {1};
    fill(target_data, target);

    tensor_t grad({1, 1, n, 1});

    ll.loss_gradient(output, target, grad);

    vec_t correct_data = {0.2, -0.7, 0.5};
    auto delta_i       = grad.host_begin();
    for (const auto& d : correct_data) {
      ASSERT_EQ(d, *delta_i++);
    }
  }

  TEST(Loss, loss) {
    size_t n      = 3;
    shape4d shape = {1, 1, n, 1};
    loss::Softmax ll;
    ll.set_shape(shape);

    tensor_t output({1, 1, n, 1});
    vec_t test_data = {0.2, 0.3, 0.5};
    fill(test_data, output);

    tensor_t target({1, 1, 1, 1});
    target.host_at_index(0) = 0;

    auto loss_value = ll.loss(output, target);
    ASSERT_NEAR(1.60944, loss_value, 1E-5);
  }
}  // namespace simpleCNN

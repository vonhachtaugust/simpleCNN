//
// Created by hacht on 4/17/17.
//

#pragma once

#include "../simpleCNN/simpleCNN.h"
#include "gtest/gtest.h"

namespace simpleCNN {

  TEST(Activation, softmax) {
    typedef activation::Softmax softmax;
    size_t n = 5;

    softmax s;

    tensor_t affine({1, 1, n, 1});
    vec_t affine_data = {std::log(1), std::log(2), std::log(3), std::log(4), std::log(5)};
    fill(affine_data, affine);

    tensor_t activated({1, 1, n, 1});

    s.forward_activation(affine, activated);

    float_t sum = std::accumulate(activated.host_begin(), activated.host_end(), float_t(0));
    ASSERT_NEAR(sum, 1, 1E-7);
  }

  TEST(Activation, real_softmax_example) {
    typedef activation::Softmax Activation;

    tensor_t tensor({1, 1, 10, 1});
    tensor_t tensor_s({1, 1, 10, 1});

    // data unrealistic from a failed initialization
    vec_t data = {-98.8815, -5.7299, 37.6746, 32.6132, 31.7426, 46.2226, -43.5877, -22.5606, -27.4484, -17.5941};

    fill(data, tensor);
    // print(tensor, "Input");

    fill(data, tensor_s);
    float_t m = max<vec_t, float_t>(data);
    tensor_s.add(-m);
    // print(tensor_s, "Input subtracted");

    Activation act;

    tensor_t a({1, 1, 10, 1});
    tensor_t a_s({1, 1, 10, 1});

    act.forward_activation(tensor, a);
    // print(a, "Activated");

    act.forward_activation(tensor_s, a_s);
    // print(a_s, "Activated subtracted");

    float_t sum  = std::accumulate(a.host_begin(), a.host_end(), float_t(0));
    float_t sum2 = std::accumulate(a_s.host_begin(), a_s.host_end(), float_t(0));
    ASSERT_NEAR(sum, 1, 2E-7);
    ASSERT_NEAR(sum2, 1, 2E-7);

    for (size_t i = 0; i < a.size(); i++) {
      ASSERT_NEAR(a.host_at_index(i), a_s.host_at_index(i), 2E-7);
    }
  }
  TEST(Activation, relu) {
    typedef activation::ReLU relu;

    vec_t test_data{1, -1, 0, -5, -3, 0, -1};
    tensor_t forward({1, 1, test_data.size(), 1});
    tensor_t backward({1, 1, test_data.size(), 1});

    fill(test_data, forward);
    fill(test_data, backward);

    tensor_t forward_a(forward.shape_v());
    tensor_t backward_a(backward.shape_v());

    tensor_t curr_delta(backward.shape_v());
    curr_delta.fill(1.0f);

    relu r;
    r.set_in_shape({1, 1, test_data.size(), 1});

    r.forward_activation(forward, forward_a);
    r.backward_activation(backward, curr_delta, backward_a);

    print(forward_a);
    print(backward_a);

    vec_t corr = {1, 0, 0, 0, 0, 0, 0};
    for (size_t i = 0; i < forward_a.size(); ++i) {
      ASSERT_EQ(forward_a.host_at_index(i), corr[i]);
      ASSERT_EQ(backward_a.host_at_index(i), corr[i]);
    }
  }
}
//
// Created by hacht on 4/17/17.
//

#pragma once

#include "../simpleCNN/simpleCNN.h"
#include "gtest/gtest.h"

namespace simpleCNN {

  TEST(Activation, softmax) {
    typedef activation::Softmax<float_t> Activation;
    size_t n = 5;

    Activation h;

    tensor_t affine({1, 1, n, 1});
    vec_t affine_data = {std::log(1), std::log(2), std::log(3), std::log(4), std::log(5)};
    fill(affine_data, affine);

    tensor_t activated({1, 1, n, 1});

    h.activate(affine, activated);

    float_t sum = std::accumulate(activated.host_begin(), activated.host_end(), float_t(0));
    ASSERT_NEAR(sum, 1, 1E-7);
  }

  TEST(Activation, real_softmax_example) {
  typedef activation::Softmax<float_t> Activation;

  tensor_t tensor({1, 1, 10, 1});
  vec_t data = { -98.8815, -5.7299, 37.6746, 32.6132, 31.7426, 46.2226 , -43.5877 , -22.5606 , -27.4484 , -17.5941};

  float_t m = max<vec_t, float_t>(data);


  fill(data, tensor);
  print(tensor, "Input");

  Activation act;

  tensor_t a({1, 1, 10, 1});
  act.activate(tensor, a);
  print(a, "Activated");

}

}
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

    float_t result = loss::regularization<float_t>(weight);
    ASSERT_EQ(result, 28);
  }

  TEST(Loss, log_likelihod) {
    typedef loss::Log_likelihood<float_t> Loss;
    typedef activation::Softmax<float_t> Soft;
    size_t n = 3;
    Soft s;

    tensor_t non_activ({1, 1, n, 1});
    vec_t non_act_data = {std::log(1), std::log(2), std::log(3)};
    fill(non_act_data, non_activ);

    tensor_t activated({1, 1, n, 1});
    s.activate(non_activ, activated);

    tensor_t target_I({1, 1, 1, 1});
    vec_t target_data_I = {0};
    fill(target_data_I, target_I);

    tensor_t target_II({1, 1, 1, 1});
    vec_t target_data_II = {1};
    fill(target_data_II, target_II);

    tensor_t target_III({1, 1, 1, 1});
    vec_t target_data_III = {2};
    fill(target_data_III, target_III);

    ASSERT_NEAR(Loss::L(activated, target_I, 1), 1.7917594, 1E-7);
    ASSERT_NEAR(Loss::L(activated, target_II, 1), 1.0986122, 2E-7);  // 1.08 E-7 ...
    ASSERT_NEAR(Loss::L(activated, target_III, 1), 0.6931471, 1E-7);
  }

  TEST(Loss, delta_log_likelihood) {
    typedef loss::Log_likelihood<float_t> Loss;
    typedef activation::Softmax<float_t> Soft;
    size_t n = 3;
    Soft s;

    tensor_t non_activ({1, 1, n, 1});
    vec_t non_act_data = {std::log(1), std::log(2), std::log(3)};
    fill(non_act_data, non_activ);

    tensor_t activated({1, 1, n, 1});
    s.activate(non_activ, activated);

    tensor_t target({1, 1, 1, 1});
    target.host_at(0, 0, 0, 0) = 0;

    tensor_t deltas({1, 1, n, 1});
    Loss::dL(activated, target, deltas, 1);

    auto activated_i = activated.host_begin();
    auto deltas_i    = deltas.host_begin();

    for (size_t i = 0; i < n; ++i) {
      if (i == 0) {
        ASSERT_NEAR(*activated_i++, *deltas_i++ + float_t(1), 1E-7);
        continue;
      }
      ASSERT_EQ(*activated_i++, *deltas_i++);
    }
  }

  TEST(Loss, gradient_loss) {
    typedef loss::Log_likelihood<float_t> Loss;
    size_t output_dim       = 3;
    const size_t batch_size = 1;

    tensor_t output({batch_size, 1, output_dim, 1});
    vec_t output_data = {0.2, 0.3, 0.5};
    fill(output_data, output);

    tensor_t target({batch_size, 1, 1, 1});
    vec_t target_data = {1};
    fill(target_data, target);

    tensor_t output_delta({batch_size, 1, output_dim, 1});

    gradient<Loss>(output, target, output_delta, batch_size);

    vec_t df_data = {0.2, -0.7, 0.5};
    auto delta_i  = output_delta.host_begin();
    for (const auto& d : df_data) {
      ASSERT_EQ(d, *delta_i++);
    }
  }

}  // namespace simpleCNN

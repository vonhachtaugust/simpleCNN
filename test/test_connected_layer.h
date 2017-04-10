//
// Created by hacht on 4/6/17.
//

#pragma once

#include "../simpleCNN/simpleCNN.h"
#include "gtest/gtest.h"

namespace simpleCNN {

  TEST(Connected, forward_propagation_raw) {
  size_t in_dim = 5;
  size_t out_dim = 3;
  size_t batch_size = 1;

  // INPUT
  tensor_t in({batch_size, 1, 1, in_dim});
  vec_t in_data = {1, 2, 3, 4, 5};
  fill(in_data, in);

  // WEIGHT
  tensor_t weight({batch_size, 1, out_dim, in_dim});
  vec_t w_data = {1, 1, 1, 1, 1,
                  -1, -1, -1, -1, -1,
                  1, 1, 1, 1, 1};
  fill(w_data, weight);

  // BIAS
  tensor_t bias({batch_size, 1, 1, out_dim});
  vec_t b_data = {-1, -1, -1};
  fill(b_data, bias);

  // OUTPUT
  tensor_t out({batch_size, 1, out_dim, 1});

  matrix_t mRows({out_dim, in_dim});
  matrix_t mCols({1, in_dim});
  matrix_t mResult({out_dim, 1});

  im2mat_cpu(weight, mRows, 0, out_dim, in_dim);
  //print(mRows, "Matrix rows");

  im2mat_cpu(in, mCols, 0, 1, in_dim);
  //print(mCols, "Matrix columns");

  sgemm(mRows, mCols, mResult, false, true);
  //print(mResult, "Matrix result");

  mat2im_cpu(mResult, out, 0, out_dim, 1);
  //print(out, "Output");

  auto iter = out.host_begin();
  vec_t correctOutput = {15, -15, 15};
  for (const auto &co : correctOutput) {
    ASSERT_EQ(*iter++, co);
  }

}
  TEST(Connected, backward_propagation_raw) {
  size_t in_dim = 4;
  size_t out_dim = 2;
  size_t batch_size = 1;

  // Current gradients
  tensor_t curr_grad({batch_size, 1, out_dim, 1});
  vec_t curr_grad_data = {1, 2};
  fill(curr_grad_data, curr_grad);

  // Previous gradients
  tensor_t prev_grad({batch_size, 1, in_dim, 1});

  // Weights
  tensor_t weight({batch_size, 1, out_dim, in_dim});
  vec_t w_data = {1, 1, 1, 1, 1,
                  -1, -1, -1, -1, -1};
  fill(w_data, weight);

  // Bias
  tensor_t bias({batch_size, 1, 1, out_dim});
  vec_t b_data = {1, -1,};
  fill(b_data, bias);

  matrix_t mRows({out_dim, in_dim});
  matrix_t mCols({out_dim, 1});
  matrix_t mResult({mRows.cols(), mCols.cols()});

  im2mat_cpu(weight, mRows, 0, out_dim, in_dim);
  //print(mRows, "Weigths as matrix: ");

  im2mat_cpu(curr_grad, mCols, 0, out_dim, 1);
  //print(mCols, "Current gradients as matrix: ");

  sgemm(mRows, mCols, mResult, true, false);
  mat2im_cpu(mResult, prev_grad, 0, in_dim, 1);
  //print(prev_grad, "Output");
  }

  TEST(Connected, forward_op) {
  size_t in_dim = 4;
  size_t out_dim = 2;
  size_t batch_size = 1;

  Connected_layer<> con(in_dim, out_dim, batch_size);

  vec_t weight_data = {1, 1, 1, 1, -1, -1, -1, -1};
  vec_t bias_data = {1, -1};

  weight_init::Test weight(weight_data);
  weight_init::Test bias(bias_data);

  con.weight_init(weight);
  con.bias_init(bias);
  con.init_weight();

  tensor_t in({batch_size, 1, in_dim, 1});
  vec_t in_data = { 1, 2, -2, -1};
  fill(in_data, in);

  tensor_t out({batch_size, 1, out_dim, 1});

  data_ptrs_t input = {&in, con.in_component(component_t::WEIGHT), con.in_component(component_t::BIAS)};
  data_ptrs_t output = {&out, &out};

  con.forward_propagation(input, output);
  //print(out, "Output");

  auto iter = out.host_begin();
  for (const auto& d : bias_data)
  {
    ASSERT_EQ(*iter++, d);
  }
}

TEST(Connected, backprop_op) {
  size_t in_dim = 4;
  size_t out_dim = 2;
  size_t batch_size = 1;

  Connected_layer<> con(in_dim, out_dim, batch_size);

  // Input
  tensor_t in({batch_size, 1, in_dim, 1});
  vec_t in_data = { 1, 2, -2, -1};
  fill(in_data, in);

  // Weight, bias
  vec_t weight_data = {1, 1, 1, 1, -1, -1, -1, -1};
  vec_t bias_data = {1, -1};

  weight_init::Test weight(weight_data);
  weight_init::Test bias(bias_data);

  con.weight_init(weight);
  con.bias_init(bias);
  con.init_weight();

  tensor_t dW({batch_size, 1, in_dim, out_dim});
  tensor_t db({batch_size, 1, out_dim, 1});

  // curr grad
  tensor_t curr_grad({batch_size, 1, out_dim, 1});
  vec_t curr_grad_data = { 1, -1 };
  fill(curr_grad_data, curr_grad);

  // prev grad
  tensor_t prev_grad({batch_size, 1, in_dim, 1});

  data_ptrs_t input = {&in, con.in_component(component_t::WEIGHT), con.in_component(component_t::BIAS)};
  data_ptrs_t output = {};
  data_ptrs_t in_grad = {&prev_grad, &dW, &db};
  data_ptrs_t out_grad = {&curr_grad, &curr_grad};

  con.back_propagation(input, output, in_grad, out_grad);

  //print(*in_grad[0], "Previous gradients");
  //print(*in_grad[1], "dW");
  //print(*in_grad[2], "db");

  vec_t cdw = {1, -1, 2, -2, -2, 2, -1, 1};
  auto iter = dW.host_begin();

  for (const auto& w : cdw)
  {
    ASSERT_EQ(*iter++, w);
  }

  vec_t cdb = {1, -1};
  auto biter = db.host_begin();
  for (const auto& b : cdb )
  {
    ASSERT_EQ(*biter++, b);
  }
}

} // namespace simpleCNN
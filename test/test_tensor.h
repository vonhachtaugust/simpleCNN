//
// Created by hacht on 4/6/17.
//

#pragma once

#include "../simpleCNN/simpleCNN.h"
#include "gtest/gtest.h"

namespace simpleCNN {

TEST(Tensor, resize)
{
  size_t w = 3;
  size_t h = 3;
  size_t c = 3;

  vec_t data;
  for (int i = 0; i < w * h * c; ++i) data.push_back(i);

  tensor_t tensor({1,3,3,3});
  fill(data, tensor);
  //print(tensor, "Original");

  tensor.resize({1,1,1,27});
  //print(tensor, "Resized");

  auto iter = tensor.host_begin();
  for (const auto& d : data)  { ASSERT_EQ(*iter++, d); }
}

}

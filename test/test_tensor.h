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

    ASSERT_EQ(tensor.host_at(1, 0, 2, 1), tensor.host_at_index(16));
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

  TEST(Tensor, subview_I) {
    size_t batch_size = 3;
    size_t in_h       = 5;
    size_t in_w       = 5;

    tensor_t total({batch_size, 1, 5, 5});
    for (size_t i = 0; i < total.size(); ++i) total.host_at_index(i) = i + 1;
    // print(total, "Total");

    auto subview = total.subView({1}, {1, 1, 5, 5});
    // print(subview, "Subview");
  }

  TEST(Tensor, subview_II) {
    size_t total_num = 10;
    size_t in_h      = 5;
    size_t in_w      = 5;

    tensor_t total({total_num, 1, in_h, in_w});

    for (size_t i = 0; i < total.size(); ++i) total.host_at_index(i) = i + 1;

    size_t batch_size = 3;

    for (size_t j = 0; j < total_num; j += batch_size) {
      auto minibatch = total.subView({j}, {batch_size, 1, in_h, in_w});
    }
  }

  TEST(Tensor, split_training_validation) {
    size_t num_images = 10;
    size_t in_h       = 3;
    size_t in_w       = 3;
    float_t ratio     = 0.8;

    tensor_t images({num_images, 1, in_h, in_w});
    tensor_t labels({num_images, 1, 1, 1});

    for (size_t i = 0; i < num_images; ++i) {
      for (size_t j = 0; j < in_h * in_w; ++j) {
        size_t index = i * in_h * in_w + j;

        images.host_at_index(index) = i * in_h * in_w + j + 1;
      }
      labels.host_at_index(i) = i + 1;
    }

    size_t training_size = num_images * ratio;
    size_t valid_size    = std::floor(num_images * (1 - ratio) + 0.5);

    tensor_t train_images({training_size, 1, in_h, in_w});
    tensor_t train_labels({training_size, 1, 1, 1});
    tensor_t valid_images({valid_size, 1, in_h, in_w});
    tensor_t valid_labels({valid_size, 1, 1, 1});

    split_training_validation(images, train_images, valid_images, ratio);
    split_training_validation(labels, train_labels, valid_labels, ratio);

    vec_t train_data;
    for (size_t i = 0; i < training_size * in_h * in_w; ++i) {
      train_data.push_back(i + 1);
    }

    vec_t valid_data;
    for (size_t i = training_size * in_h * in_w; i < num_images; ++i) {
      valid_data.push_back(i + 1);
    }

    vec_t train_label_data;
    for (size_t i = 0; i < training_size; ++i) {
      train_label_data.push_back(i + 1);
    }

    vec_t valid_label_data;
    for (size_t i = training_size; i < num_images; ++i) {
      valid_label_data.push_back(i + 1);
    }

    auto i = train_images.host_begin();
    for (auto d : train_data) {
      ASSERT_EQ(d, *i++);
    }

    auto it = valid_images.host_begin();
    for (auto d : valid_data) {
      ASSERT_EQ(d, *it++);
    }

    auto ite = train_labels.host_begin();
    for (auto d : train_label_data) {
      ASSERT_EQ(d, *ite++);
    }

    auto iter = valid_labels.host_begin();
    for (auto d : valid_label_data) {
      ASSERT_EQ(d, *iter++);
    }
  }
}

//
// Created by hacht on 4/28/17.
//

#pragma once

namespace simpleCNN {


template<typename Container, typename T>
T mean_value(const Container& c, const size_t container_size) {
  std::accumulate(std::begin(c), std::end(c), T(0)) / static_cast<T>(container_size);
}

template<typename Container, typename T>
void add(Container& c, const T value) {
  for (auto iter = std::begin(c); iter != std::end(c); ++iter) {
    *iter += value;
  }
};

template<typename Container, typename T>
T max(const Container& c) {
  T max = T(0);
  for (auto iter = std::begin(c); iter != std::end(c); ++iter) { if (*iter > max) { max = *iter; }}
  return max;
};

float_t mean_value(const tensor_t& x) {
  return std::accumulate(x.host_begin(), x.host_end(), float_t{0}) / static_cast<float_t>(x.size());
}

void mean(tensor_t& x, const size_t batch_size) {
  float_t norm = float_t(1) / float_t(batch_size);

  for (auto iter = x.host_begin(); iter != x.host_end(); ++iter) {
    *iter *= norm;
  }
}

void mean_and_regularize(const tensor_t& x, tensor_t& dx, const size_t batch_size) {
  float_t norm = float_t(1) / float_t(batch_size);
  float_t reg = float_t(0.001);

  size_t n = dx.size();

  for(size_t i = 0; i < n; ++i) {
    dx.host_at_index(i) = (dx.host_at_index(i) + reg * x.host_at_index(i)) * norm;
  }
}


}
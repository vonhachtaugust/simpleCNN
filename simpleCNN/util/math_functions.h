//
// Created by hacht on 4/28/17.
//

#pragma once

#include "../io/serialize.h"

namespace simpleCNN {

  template <typename Container, typename T>
  T mean_value(const Container& c, const size_t container_size) {
    std::accumulate(std::begin(c), std::end(c), T(0)) / static_cast<T>(container_size);
  }

  template <typename Container, typename T>
  void add(Container& c, const T value) {
    for (auto iter = std::begin(c); iter != std::end(c); ++iter) {
      *iter += value;
    }
  };

  template <typename Container, typename T>
  T max(const Container& c) {
    T max = T(0);
    for (auto iter = std::begin(c); iter != std::end(c); ++iter) {
      if (*iter > max) {
        max = *iter;
      }
    }
    return max;
  };

  std::vector<float_t> standard_deviation(const tensor_t& x, const std::vector<float_t> means) {
    size_t batch_size   = x.shape()[0];
    size_t batch_length = x.size() / batch_size;

    std::vector<float_t> stds;

    for (size_t b = 0; b < batch_size; ++b) {
      float_t std = float_t(0);
      for (size_t i = 0; i < batch_length; ++i) {
        size_t index = b * batch_length + i;

        std += std::pow(x.host_at_index(index) - means[b], 2);
      }
      stds.push_back(std::sqrt(std / static_cast<float_t>(batch_length)));
    }

    return stds;
  }

  std::vector<float_t> means(const tensor_t& x) {
    size_t batch_size   = x.shape()[0];
    size_t batch_length = x.size() / batch_size;

    std::vector<float_t> means;

    for (size_t b = 0; b < batch_size; ++b) {
      float_t mean = float_t(0);
      for (size_t i = 0; i < batch_length; ++i) {
        size_t index = b * batch_length + i;

        mean += x.host_at_index(index);
      }
      means.push_back(mean / static_cast<float_t>(batch_length));
    }

    return means;
  }

  void zero_mean(tensor_t& x, const float_t mean) {
    size_t batch_size   = x.shape()[0];
    size_t batch_length = x.size() / batch_size;

    for (size_t b = 0; b < batch_size; ++b) {
      for (size_t i = 0; i < batch_length; ++i) {
        size_t index = b * batch_length + i;

        x.host_at_index(index) -= mean;
      }
    }
  }

  std::vector<float_t> zero_mean(tensor_t& x) {
    std::vector<float_t> result;
    auto m = means(x);
    float_t mean = std::accumulate(m.begin(), m.end(), float_t(0)) / static_cast<float_t>(m.size());

    size_t batch_size   = x.shape()[0];
    size_t batch_length = x.size() / batch_size;

    for (size_t b = 0; b < batch_size; ++b) {
      for (size_t i = 0; i < batch_length; ++i) {
        size_t index = b * batch_length + i;

        x.host_at_index(index) -= mean;
      }
    }
    result.push_back(mean);
    return result;
  }

  void zero_mean_unit_variance(tensor_t& x, const float_t& mean, const float_t& std) {
    size_t batch_size   = x.shape()[0];
    size_t batch_length = x.size() / batch_size;

    for (size_t b = 0; b < batch_size; ++b) {
      for (size_t i = 0; i < batch_length; ++i) {
        size_t index = b * batch_length + i;

        x.host_at_index(index) -= mean;
        x.host_at_index(index) /= std;
      }
    }
  }

  std::vector<float_t> zero_mean_unit_variance(tensor_t& x) {
    std::vector<float_t> result;
    auto m = means(x);
    auto s = standard_deviation(x, m);

    float_t mean = std::accumulate(m.begin(), m.end(), float_t(0)) / static_cast<float_t>(m.size());
    float_t std  = std::accumulate(s.begin(), s.end(), float_t(0)) / static_cast<float_t>(s.size());

    size_t batch_size   = x.shape()[0];
    size_t batch_length = x.size() / batch_size;

    for (size_t b = 0; b < batch_size; ++b) {
      for (size_t i = 0; i < batch_length; ++i) {
        size_t index = b * batch_length + i;

        x.host_at_index(index) -= mean;
        x.host_at_index(index) /= std;
      }
    }
    result.push_back(mean);
    result.push_back(std);

    return result;
  }

  void zero_mean_unit_variance(tensor_t& x, std::string filename) {
    std::vector<float_t> data(2);
    load_data_from_file<float_t>(filename, data);

    float_t mean = data[0];
    float_t std  = data[1];

    size_t batch_size   = x.shape()[0];
    size_t batch_length = x.size() / batch_size;

    for (size_t b = 0; b < batch_size; ++b) {
      for (size_t i = 0; i < batch_length; ++i) {
        size_t index = b * batch_length + i;

        x.host_at_index(index) -= mean;
        x.host_at_index(index) /= std;
      }
    }
  }

  /**
   * Regularization term. Without it solution is not guaranteed to be unique.
   *
   * @tparam T
   * @param weights
   * @return
   */
  template <typename T>
  T regularization(const std::vector<tensor_t*>& weight) {
    float_t dot_sum = float_t(0);

    for (size_t i = 0; i < weight.size(); ++i) {
      auto w = weight[i];

      dot_sum += dot(*w, &(*w->host_begin()), *w, &(*w->host_begin()));
    }

    // return dot(weight, &(*weight.host_begin()), weight, &(*weight.host_begin()));
    return dot_sum;
  }

  float_t l2norm(const tensor_t& a, const tensor_t& b) {
    float_t sum = float_t(0);

    for (size_t i = 0; i < a.size(); ++i) {
      auto val_a = a.host_at_index(i);
      auto val_b = b.host_at_index(i);

      sum += (val_a + val_b) * (val_a + val_b);
    }
    return sum;
  }

  float_t l2norm_diff(const tensor_t& a, const tensor_t& b) {
    float_t sum = float_t(0);

    for (size_t i = 0; i < a.size(); ++i) {
      auto val_a = a.host_at_index(i);
      auto val_b = b.host_at_index(i);

      sum += (val_a - val_b) * (val_a - val_b);
    }
    return sum;
  }

  std::vector<float_t> relative_error(const std::vector<tensor_t*>& A, std::vector<tensor_t>& B) {
    assert(A.size() == B.size());
    std::vector<float_t> errors;

    for (size_t i = 0; i < B.size(); ++i) {
      auto sum  = l2norm(*A[i], B[i]);
      auto diff = l2norm_diff(*A[i], B[i]);
      auto val  = diff / sum;
      if (isnan(val)) {
        if (sum == 0 && diff == 0) {
          val = 0;
          continue;
        }
        throw simple_error("Error: sum or diff was nan");
      }

      errors.push_back(val);
    }
    return errors;
  }
}
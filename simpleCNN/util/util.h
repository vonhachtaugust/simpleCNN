//
// Created by hacht on 3/4/17.
//

#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>

#include "../core/framework/tensor.h"

#ifdef USE_OPENCV
#include "opencv2/core/core.hpp"
#include "opencv2/highgui.hpp"
#endif


namespace simpleCNN {

  enum class net_phase { train, test };

  struct Hyperparameters {
    constexpr static float_t regularization_constant = 1E-3;
    constexpr static float_t learning_rate           = 5E-4;

    //constexpr static float_t regularization_constant = 1E-3;
    //constexpr static float_t learning_rate           = 1E-4;
  };

  template <typename T = float_t, typename Allocator = aligned_allocator<T, 64>>
  using default_vec_t = std::vector<T, Allocator>;

  using vec_t      = default_vec_t<>;
  using vec_iter_t = vec_t::iterator;

  template <typename T = float_t, bool kConst = false, typename Allocator = aligned_allocator<T, 64>>
  using default_array_t = Tensor<T, 1, kConst, Allocator>;
  using array_t         = default_array_t<>;

  /* ------------------------------------------------------------------- //
   * Matrix type used in matrix multiplications
   *
   * note: float_t important since it is the supported precision
   * by the matrix convolution kernel operations
   * (double precision exists also).
   *
   */
  template <typename T = float_t, bool kConst = false, typename Allocator = aligned_allocator<T, 64>>
  using default_matrix_t = Tensor<T, 2, kConst, Allocator>;
  using matrix_t         = default_matrix_t<>;  // no more <> yey
  using matrix_data_t    = std::vector<matrix_t>;
  using vec_matrix_ptr_t = std::vector<matrix_t *>;

  /* ------------------------------------------------------------------- //
   *
   * note: float_t important since it is the supported precision
   * by the matrix convolution kernel operations
   * (double precision exists also).
   *
   */
  template <typename T = float_t, bool kConst = false, typename Allocator = aligned_allocator<T, 64>>
  using default_tensor_t = Tensor<T, 4, kConst, Allocator>;

  using tensor_t     = default_tensor_t<>;  // removed <>  yey
  using tensor_ptr_t = std::shared_ptr<tensor_t>;
  // using vec_tensor_ptr_t = std::vector<tensor_t *>;

  using data_t      = std::vector<tensor_t>;
  using data_ptrs_t = std::vector<tensor_t *>;
  // ------------------------------------------------------------------- //

  using shape4d = std::vector<size_t>;
  using shape_t = std::vector<shape4d>;

  /* ------------------------------------------------------------------- //
   * Constructor arguments require data type composition for initialization.
   * Component type important and used frequently.
   */
  inline data_t std_input_order(bool has_bias) {
    if (has_bias) {
      return {tensor_t(component_t::IN_DATA), tensor_t(component_t::WEIGHT), tensor_t(component_t::BIAS)};
    } else {
      return {tensor_t(component_t::IN_DATA), tensor_t(component_t::WEIGHT)};
    }
  }

  inline data_t std_output_order(bool has_activation) {
    if (has_activation) {
      return {tensor_t(component_t::OUT_DATA), tensor_t(component_t::AUX)};
    } else {
      return {tensor_t(component_t::OUT_DATA)};
    }
  }
  // ------------------------------------------------------------------- //

  inline bool is_high_endian() {
    union {
      uint32_t i;
      char c[4];
    } test_endian = {0x01020304};

    return test_endian.c[0] == 1;
  }

  uint32_t swap_uint32(uint32_t val) {
    val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
    return (val << 16) | (val >> 16);
  }

  std::string get_time_stamp() {
    std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();
    std::time_t time                                         = std::chrono::system_clock::to_time_t(start);
    std::string d                                            = std::ctime(&time);
    std::string c                                            = d.substr(0, d.size() - 1);

    std::replace_if(c.begin(), c.end(), isspace, '_');
    return c;
  }

  std::vector<size_t> histogram(const tensor_t& x, const size_t& num_classes) {
    std::vector<size_t> hist(num_classes, 0);

    for (auto iter = x.host_begin(); iter != x.host_end(); ++iter) {
      hist[*iter]++;
    }

    return hist;
  }

#ifdef USE_OPENCV
static void display_image(const tensor_t& test_image, const size_t label, const float_t mean, const float_t std) {
  cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE);

  size_t in_ch = test_image.dimension(dim_t::depth);
  size_t in_height = test_image.dimension(dim_t::height);
  size_t in_width = test_image.dimension(dim_t::width);

  cv::Mat image(in_height, in_width, CV_8UC3);  // rgb

  for (size_t ch = 0; ch < in_ch; ++ch) {
    for (size_t i = 0; i < in_height; ++i) {
      for (size_t j = 0; j < in_width; ++j) {
        auto val = test_image.host_at(0, ch, i, j);
        val = (val*std + mean + 1) * (255.0f / 2.0f);
        image.at<cv::Vec3b>(i, j)[in_ch - ch - 1] = val;
      }
    }
  }
  std::cout << "Label: " << label << std::endl;
  cv::imshow("Display window", image);
  cv::waitKey(0);
}

static void display_gray_image(const tensor_t& test_image, const size_t label, const float_t mean, const float_t std) {
  cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE);

  size_t in_height = test_image.dimension(dim_t::height);
  size_t in_width = test_image.dimension(dim_t::width);

  cv::Mat image(in_height, in_width, CV_8UC1);  // gray
  uchar* p = image.data;
    for (size_t i = 0; i < in_height; ++i) {
      for (size_t j = 0; j < in_width; ++j) {
        auto val = test_image.host_at(0, 0, i, j);
        val = (val*std + mean + 1) * (255.0f / 2.0f);
        p[i * in_width + j] = val;
      }
    }
  std::cout << "Label: " << label << std::endl;
  cv::imshow("Display window", image);
  cv::waitKey(0);
}
#endif

}  // namespace simpleCNN

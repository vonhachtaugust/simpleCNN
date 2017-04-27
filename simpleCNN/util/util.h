//
// Created by hacht on 3/4/17.
//

#pragma once

#include "../core/framework/tensor.h"

namespace simpleCNN {
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
      return data_t({tensor_t(component_t::IN_DATA), tensor_t(component_t::WEIGHT), tensor_t(component_t::BIAS)});
    } else {
      return data_t({tensor_t(component_t::IN_DATA), tensor_t(component_t::WEIGHT)});
    }
  }

  inline data_t std_output_order(bool has_activation) {
    if (has_activation) {
      return data_t({tensor_t(component_t::OUT_DATA), tensor_t(component_t::AUX)});
    } else {
      return data_t({tensor_t(component_t::OUT_DATA)});
    }
  }
  // ------------------------------------------------------------------- //

  inline bool is_high_endian() {
      union {
        uint32_t i;
        char c[4];
      } test_endian = { 0x01020304 };

    return test_endian.c[0] == 1;
  }

  uint32_t swap_uint32( uint32_t val )
  {
  val = ((val << 8) & 0xFF00FF00 ) | ((val >> 8) & 0xFF00FF );
  return (val << 16) | (val >> 16);
  }

}  // namespace simpleCNN

//
// Created by hacht on 2/24/17.
//

#pragma once

#include <algorithm>  // std::fill, std::generate
#include <cmath>      // sqrt
#include <numeric>    // std::accumulate
#include <vector>

#include "../../util/util.h"
#include "cblas.h"
#include "tensor.h"

namespace simpleCNN {

  template <typename T>
  void print(T t) {
    std::cout << t << std::endl;
  }

  template<typename T>
  void print_seq(T t) {
    std::cout << t << "\t";
  }

template<typename Container>
void printc(Container c, const std::string& name) {
  simple_info(name.c_str());
  for (auto iter = std::begin(c); iter != std::end(c); ++iter) {
    print_seq(*iter);
  }
  std::cout << std::endl;
}

  template <typename T>
  void print(T t, const std::string& name) {
    simple_info(name.c_str());
    print(t);
  }

  template <typename T>
  void print_pack(std::ostream& out, T t) {
    out << t;
  }

  template <typename T, typename U, typename... Args>
  void print_pack(std::ostream& out, T t, U u, Args... args) {
    out << t << ',';
    print_pack(out, u, args...);
  }

  template <typename T, size_t kDim, typename... Args>
  inline std::ostream& print_last_two_dimensions(std::ostream& os, const Tensor<T, kDim>& tensor, const Args... args) {
    const std::array<size_t, kDim>& shape = tensor.shape();
    for (size_t k = 0; k < shape[kDim - 2]; ++k) {
      for (size_t l = 0; l < shape[kDim - 1]; ++l) {
        os << "\t" << tensor.host_at(args..., k, l);
      }
      os << "\n";
    }
    return os;
  }

  template <typename T,
            size_t kDim,
            typename... Args,
            typename std::enable_if<sizeof...(Args) == kDim - 3, int>::type = 0>
  inline std::ostream& print_last_n_dimensions(std::ostream& os,
                                               const Tensor<T, kDim>& tensor,
                                               const int d,
                                               const Args... args) {
    os << "Tensor(";
    print_pack(os, d, args...);
    os << ",:,:):\n";
    print_last_two_dimensions(os, tensor, d, args...);
    return os;
  }

  /*template<typename T,
      size_t kDim,
      typename std::enable_if<(kDim == 2), int>::type = 0>
  inline std::ostream& print_last_n_dimensions(std::ostream& os, const Tensor<T, kDim>& tensor, int d)
  {
    const std::array<size_t, kDim>& shape = tensor.shape();
    for (size_t i = 0; i < shape[kDim - 2]; ++i) {
      os << "\t" << tensor.host_at(d, i);
    }
    os << "\n";
    return os;
  };*/

  template <typename T,
            size_t kDim,
            typename... Args,
            typename std::enable_if<(sizeof...(Args) < kDim - 3), int>::type = 0>
  inline std::ostream& print_last_n_dimensions(std::ostream& os,
                                               const Tensor<T, kDim>& tensor,
                                               const int d,
                                               const Args... args) {
    const std::array<size_t, kDim>& shape = tensor.shape();
    const size_t n_dim = sizeof...(args);
    for (size_t k = 0; k < shape[n_dim + 1]; ++k) {
      print_last_n_dimensions(os, tensor, d, args..., k);
    }
    return os;
  }

  template <typename T>
  inline std::ostream& operator<<(std::ostream& os, const Tensor<T, 1>& tensor) {
    for (auto iter = tensor.host_begin(); iter != tensor.host_end(); ++iter) {
      os << "\t" << *iter;
    }
    os << "\n";
    return os;
  }

  template <typename T>
  inline std::ostream& operator<<(std::ostream& os, const Tensor<T, 2>& tensor) {
    print_last_two_dimensions(os, tensor);
    return os;
  }

  template <typename T, size_t kDim>
  inline std::ostream& operator<<(std::ostream& os, const Tensor<T, kDim>& tensor) {
    const std::array<size_t, kDim>& shape = tensor.shape();
    for (size_t i = 0; i < shape[0]; ++i) print_last_n_dimensions(os, tensor, i);
    return os;
  }

  template <typename T, size_t kDim, typename... Args>
  inline void fill_last_two_dimensions(vec_iter_t& curr,
                                       Tensor<T, kDim>& tensor,
                                       const vec_iter_t& end,
                                       const Args... args) {
    const std::array<size_t, kDim>& shape = tensor.shape();
    for (size_t k = 0; k < shape[kDim - 2]; ++k) {
      for (size_t l = 0; l < shape[kDim - 1] && curr != end; ++l) {
        *tensor.host_iter(args..., k, l) = *curr++;
      }
    }
  };

  template <typename T,
            size_t kDim,
            typename... Args,
            typename std::enable_if<(sizeof...(Args) == kDim - 3), int>::type = 0>
  inline void fill_last_n_dimensions(
    vec_iter_t& curr, Tensor<T, kDim>& tensor, const vec_iter_t& end, const int d, const Args... args) {
    fill_last_two_dimensions(curr, tensor, end, d, args...);
  };

  template <typename T,
            size_t kDim,
            typename... Args,
            typename std::enable_if<(sizeof...(Args) < kDim - 3), int>::type = 0>
  inline void fill_last_n_dimensions(
    vec_iter_t& curr, Tensor<T, kDim>& tensor, const vec_iter_t& end, const int d, const Args... args) {
    const std::array<size_t, kDim>& shape = tensor.shape();
    const size_t n_dim = sizeof...(args);
    for (size_t k = 0; k < shape[n_dim + 1]; ++k) {
      fill_last_n_dimensions(curr, tensor, end, d, args..., k);
    }
  };

  template <typename T, size_t kDim>
  inline void fill(vec_t& data, Tensor<T, kDim>& tensor) {
    vec_iter_t curr      = data.begin();
    const vec_iter_t end = data.end();
    const std::array<size_t, kDim>& shape = tensor.shape();
    for (size_t i = 0; i < shape[0]; ++i) {
      fill_last_n_dimensions(curr, tensor, end, i);
    }
  }

  template <typename T, size_t kDim>
  inline void fill(vec_t& data, Tensor<T, kDim>& tensor, const std::string& name) {
    fill(data, tensor);
    print(tensor, name.c_str());
  };

  template <typename T>
  inline void fill(vec_t& data, Tensor<T, 2>& tensor) {
    vec_iter_t curr      = data.begin();
    const vec_iter_t end = data.end();
    fill_last_two_dimensions(curr, tensor, end);
  }

  template <typename T>
  inline void fill(vec_t& data, Tensor<T, 2>& tensor, const std::string& name) {
    fill(data, tensor);
    print(tensor, name.c_str());
  }

  template <typename T>
  inline void fill(vec_t& data, Tensor<T, 1>& tensor) {
    vec_iter_t curr      = data.begin();
    const vec_iter_t end = data.end();
    const std::array<size_t, 1>& shape = tensor.shape();
    for (size_t i = 0; i < shape[0] && curr != end; ++i) {
      *tensor.host_iter(i) = *curr++;
    }
  }

  /**
     * OpenBLAS efficient matrix multiplication. This version supports
     * floating point precision only, double precision also exists but is not going to be used here.
     * The wrapper calling function is right below this function.
     *
     */
  static void matrix_multiplcation(
    float* A, int A_width, int A_height, float* B, int B_width, int B_height, float* AB, bool tA, bool tB, float beta) {
    int A_height_new = tA ? A_width : A_height;
    int A_width_new  = tA ? A_height : A_width;
    int B_height_new = tB ? B_width : B_height;
    int B_width_new  = tB ? B_height : B_width;
    int m            = A_height_new;
    int n            = B_width_new;
    int k            = A_width_new;

    // Error, width and height should match!
    assert(A_width_new == B_height_new);
    int lda = tA ? m : k;
    int ldb = tB ? k : n;
#define TRANSPOSE(X) ((X) ? CblasTrans : CblasNoTrans)
    // http://www.netlib.org/lapack/explore-html/d7/d2b/sgemm_8f.html
    cblas_sgemm(CblasRowMajor,  // I know you
                TRANSPOSE(tA), TRANSPOSE(tB), m, n, k, 1.0f, A, lda, B, ldb, beta, AB, n);
#undef TRANSPOSE(X)
  }

  /**
   * Use this for matrix_t (float_t) multiplication
   *
   */
  static void sgemm(matrix_t& A, matrix_t& B, matrix_t& AB, bool transpose_A, bool transpose_B) {
    matrix_multiplcation(&A.host_at(0, 0), A.shape()[1], A.shape()[0], &B.host_at(0, 0), B.shape()[1], B.shape()[0],
                         &AB.host_at(0, 0), transpose_A, transpose_B, 0);
  }

  /**
   * Float matrix multiplication of two tensors.
   *
   * @tparam kDims              tensor dimensions, default 4
   * @tparam kConst             if tensor is accesses as const or not
   * @tparam Allocator          allocator type used by iterator to determine next
   * @param A                   left operand
   * @param start_A             start location in tensor A, should be A.host_iter(batch_i, channel_i, 0, 0) for default
   * tensor
   * @param B                   right operand
   * @param start_B             start location in tensor B, should be B.host_iter(batch_i, channel_i, 0, 0) for default
   * tensor
   * @param AB                  matrix multiplication result
   * @param start_AB            start location in tensor AB, should be AB.host_iter(batch_i, channel_i, 0, 0) for
   * default tensor
   * @param transpose_A         if A needs to be transposed (index iterated backwards)
   * @param transpose_B         if B needs to be transposed (index iterated backwards)
   */
  template <size_t kDims = 4, bool kConst = false, typename Allocator = aligned_allocator<float_t, 64>>
  static void multiply(const Tensor<float_t, kDims, kConst, Allocator>& A,
                       float_t* start_A,
                       const Tensor<float_t, kDims, kConst, Allocator>& B,
                       float_t* start_B,
                       float_t* start_AB,
                       bool transpose_A,
                       bool transpose_B) {
    int width_A  = A.shape()[kDims - 1];
    int height_A = A.shape()[kDims - 2];
    int width_B  = B.shape()[kDims - 1];
    int height_B = B.shape()[kDims - 2];
    matrix_multiplcation(start_A, width_A, height_A, start_B, width_B, height_B, start_AB, transpose_A, transpose_B, 0);
  }

  static float_t matrix_dot_product(const int& length, float* A, const int incm, float* B, const int incn) {
    return cblas_sdot(length, A, incm, B, incn);
  }

  template <size_t kDims = 4, bool kConst = false, typename Allocator = aligned_allocator<float_t, 64>>
  static float_t dot(const Tensor<float_t, kDims, kConst, Allocator>& A,
                     float_t* start_A,
                     const Tensor<float_t, kDims, kConst, Allocator>& B,
                     float_t* start_B) {
    int width_A  = A.shape()[kDims - 1];
    int height_A = A.shape()[kDims - 2];
    int width_B  = B.shape()[kDims - 1];
    int height_B = B.shape()[kDims - 2];

    assert(width_A == width_B);
    assert(height_A == height_B);
    assert(width_A * height_A == width_B * height_B);
    int length = width_A * height_A;

    return matrix_dot_product(length, start_A, 1, start_B, 1);
  }
}  // namespace simpleCNN

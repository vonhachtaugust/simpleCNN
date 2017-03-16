//
// Created by hacht on 2/24/17.
//

#pragma once

#include <cmath>
#include <memory>
#include <numeric>
#include <type_traits>
#include <vector>
#include "../../util/aligned_allocator.h"

namespace simpleCNN {
  template <typename Container>
  static inline size_t product(Container& c) {
    return std::accumulate(std::begin(c), std::end(c), size_t(1),
                           std::multiplies<size_t>());
  }

  template <typename C1, typename C2>
  static inline size_t compute_offset(const C1& start, const C2& shape) {
    size_t res = 0;
    for (size_t i = 0; i < shape.size(); ++i) {
      res *= shape[i];
      res += (i < start.size()) ? *(start.begin() + i) : 0;
    }
    return res;
  }

  template <typename T = float_t, typename Allocator = aligned_allocator<T, 64>>
  class TensorStorage {
    typedef typename std::vector<T, Allocator>::iterator DataIter;
    typedef typename std::vector<T, Allocator>::const_iterator ConstDataIter;

   public:
    TensorStorage(){};

    /**
     *
     * @return vector iterator over all arrays
     */
    DataIter Iterator() { return host_data_; }

    /**
     *
     * @return const vector iterator over all arrays
     */
    ConstDataIter Iterator() const { return host_data_; }

    /**
             * Constructor that accepts a vector of shape and create a
     * TensorStorage
             * with a size equivalent to that shape.
             * @param shape array containing N integers, sizes of dimensions
             * @return
             */
    explicit TensorStorage(const std::vector<size_t>& shape) { resize(shape); }

    /**
     * Constructor that accepts an initializer list  of shape and create a
     * TensorStorage with a size equivalent to that shape.
     * @param shape array containing N integers, sizes of dimensions
     * @return
     */

    /**
    *
    * @param offset
    * @return iterator to an element at offset position
    */
    DataIter host_data(size_t offset) { return host_data_.begin() + offset; }

    /**
     *
     * @param offset
     * @return  constant iterator to an element at offset position
     */
    ConstDataIter host_data(size_t offset) const {
      return host_data_.begin() + offset;
    }

    explicit TensorStorage(std::initializer_list<size_t> const& shape) {
      resize(shape);
    }

    void resize(const std::vector<size_t>& sz) {
      host_data_.resize(product(sz));
    }

    void resize(std::initializer_list<size_t> const& sz) {
      host_data_.resize(product(sz), T(0));
    }

    size_t size() const { return host_data_.size(); }

    ~TensorStorage() = default;

   private:
    /**
     *
     * Vector containing the host tensor data in the stack
     * */
    std::vector<T, Allocator> host_data_;
  };
}  // namespace simpleCNN

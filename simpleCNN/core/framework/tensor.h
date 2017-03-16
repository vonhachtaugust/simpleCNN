#pragma once

#include <cmath>
#include <memory>
#include <type_traits>
#include <vector>

#include "../../util/aligned_allocator.h"
#include "tensor_storage.h"

namespace simpleCNN {

  template <typename T         = float_t,
            size_t kDimensions = 4,
            bool kConst        = false,
            typename Allocator = aligned_allocator<T, 64>>
  class Tensor {
    // Define constant types for constant Tensor,
    // and mutable ones for mutable Tensor
    typedef typename std::conditional<kConst,
                                      const TensorStorage<T, Allocator>,
                                      TensorStorage<T, Allocator>>::type
      TensorStorageType;
    typedef typename std::conditional<kConst, const T*, T*>::type UPtr;
    typedef typename std::shared_ptr<TensorStorageType> TensorStoragePointer;
    typedef typename std::conditional<
      kConst,
      typename std::vector<T, Allocator>::const_iterator,
      typename std::vector<T, Allocator>::iterator>::type StorageIterator;

   public:
    Tensor() {
      offset_      = size_t(0);
      storage_ptr_ = std::make_shared<TensorStorageType>();
    }

    /**
    * Constructor that assepts an array of shape and create a Tensor with that
    * shape. For example, given shape = {2,3,4,5,6}, tensor
    * will be of size 2x3x4x5x6
    * @param shape array containing N integers, sizes of dimensions
    * @return
    */
    explicit Tensor(const std::array<size_t, kDimensions>& shape) {
      offset_      = size_t(0);
      shape_       = shape;
      size_        = product(shape);
      storage_ptr_ = std::make_shared<TensorStorageType>(shape);
    }

    /**
     * Constructor that assepts a vector of shape and create a Tensor with that
     * shape. For example, given shape = {2,3,4,5,6}, tensor
     * will be of size 2x3x4x5x6
     * @param shape array containing N integers, sizes of dimensions
     * @return
     */
    explicit Tensor(const std::vector<size_t>& shape) {
      offset_ = size_t(0);
      size_   = product(shape);
      std::copy(shape.begin(), shape.end(), shape_.begin());
      storage_ptr_ = std::make_shared<TensorStorageType>(shape);
    }

    /**
     * Constructor that assepts an initializer list of shape and create a
     * Tensor with that shape. For example, given shape = {2,3,4,5,6}, tensor
     * will be of size 2x3x4x5x6
     * @param shape array containing N integers, sizes of dimensions
     * @return
     */
    explicit Tensor(std::initializer_list<size_t> const& shape) {
      offset_ = size_t(0);
      size_   = product(shape);
      std::copy(shape.begin(), shape.end(), shape_.begin());
      storage_ptr_ = std::make_shared<TensorStorageType>(shape);
    }

    ~Tensor() = default;

    /**
     *
     * @return tensor shape
     */
    const std::array<size_t, kDimensions>& shape() const { return shape_; };

    /**
    * Checked version of access to indexes in tensor (throw exceptions
    * for out-of-range error)
    * @param args indexes in tensor
    * @return the value of a specified index in the tensor
    */
    template <typename... Args>
    T& host_at(const Args... args) {
      static_assert(!kConst, "Non-constant operation on constant Tensor");
      return *host_ptr(args...);
    }

    /**
     * Checked version of access to indexes in tensor (throw exceptions
     * for out-of-range error)
     * @param args indexes in tensor
     * @return the value of a specified index in the tensor
     */
    template <typename... Args>
    T host_at(const Args... args) const {
      return *host_ptr(args...);
    }

    /**
     * Calculate an offset for last dimension.
     * @param d an index of last dimension
     * @return offest from the beginning of the dimesion
     */
    size_t host_pos(const size_t d) const {
      if (d >= shape_.back()) {
        throw simple_error("Access tensor out of range.");
      }
      return d;
    }

    /**
     * Calculate an offest in 1D representation of nD Tensor. Parameters are
     * indexes of k last dimensions. If k is less than n, function returns an
     * offset from the first index of (n-k+1)th dimension. This allows recursive
     * call to acquire offset for generic number of dimensions
     * @param d index of (k-n)th dimension. For external call, n=k usually holds
     * @param args index of rest (k-1) dimensions.
     * @return offset from the first index of (n-k)th dimension
     */
    template <typename... Args>
    size_t host_pos(const size_t d, const Args... args) const {
      static_assert(sizeof...(args) < kDimensions,
                    "Wrong number of dimensions");
      size_t dim = kDimensions - sizeof...(args)-1;
      if (d >= shape_[dim]) {
        throw simple_error("Access tensor out of range.");
      }
      size_t shift = 1;
      for (size_t i = dim + 1; i < kDimensions; ++i) shift *= shape_[i];

      return (d * shift + host_pos(args...));
    }

    template <typename... Args>
    UPtr host_ptr(const Args... args) const {
      return &(*host_iter(args...));
    }

    template <typename... Args>
    StorageIterator host_iter(const Args... args) const {
      static_assert(!kConst, "Non-constant operation on constant Tensor");
      static_assert(sizeof...(args) == kDimensions,
                    "Wrong number of dimensions");
      return storage_ptr_->host_data(offset_) + host_pos(args...);
    }

    StorageIterator host_begin() const {
      return storage_ptr_->host_data(offset_);
    }

    Tensor& fill(T value) {
      static_assert(!kConst, "Non-constant operation on constant Tensor");
      // data_is_on_host_ = true;
      // data_dirty_ = true;
      std::fill(storage_ptr_->host_data(offset_),
                storage_ptr_->host_data(offset_) + size_, value);
      return *this;
    }

    void reshape(const std::array<size_t, kDimensions>& sz) {
      static_assert(!kConst, "Non-constant operation on constant Tensor");
      // No size change for reshape
      if (calcSize() != product(sz)) {
        throw simple_error("Reshape to Tensor of different size.");
      }
      shape_ = sz;
    }

    void resize(const std::array<size_t, kDimensions>& sz) {
      if (offset_ != 0 || size_ != storage_ptr_->size()) {
        throw simple_error("Resize of partial view is impossible.");
      }
      shape_ = sz;
      storage_ptr_->resize(std::vector<size_t>(sz.begin(), sz.end()));
    }

    size_t size() const { return size_; }

    Tensor operator[](size_t index) {
      std::array<size_t, kDimensions - 1> new_tensor;
      std::copy(shape_.begin() + 1, shape_.end(), new_tensor.begin());
      return Tensor(
        storage_ptr_, offset_ + index * size_ / shape_[0],
        /*std::array<size_t, kDimensions - 1>(shape_.begin() + 1, shape_.end())*/ new_tensor);
    }

    /**
    * @brief Returns a sub view from the current tensor with a given size.
    * The new tensor will share data with its parent tensor so that each time
    * that data is modified, it will be updated in both directions.
    *
    * The new sub view tensor will be extracted assuming continuous data.
    * The offset to shared data is assumed to be 0.
    *
    * @param new_shape The size for the new tensor
    * @return An instance to the new tensor
    *
    * Usage:
    *
    *  Tensor<float_t, 4> t({2,2,2,2});            // we create a 4D tensor
    *  Tensor<float_t, 4> t_view = t.view({2,2});  // we create a 2x2 matrix
    * view
    * with offset zero
    *
    */
    Tensor subView(std::initializer_list<size_t> const& new_shape) {
      return subview_impl({}, new_shape);
    }

    /**
     * @brief Returns a sub view from the current tensor with a given size.
     * The new tensor will share data with its parent tensor so that each time
     * that data is modified, it will be updated in both directions.
     *
     * The new sub view tensor will be extracted assuming continuous data.
     *
     * @param start The offset from the parent tensor
     * @param new_shape The size for the new tensor
     * @return An instance to the new tensor
     *
     * Usage:
     *
     *  Tensor<float_t, 4> t({2,2,2,2});                   // we create a 4D
     * tensor
     *  Tensor<float_t, 4> t_view = t.view({2,2}, {2,2});  // we create a 2x2
     * matrix view from
     *                                                     // offset 4.
     */
    Tensor subView(std::initializer_list<size_t> const& start,
                   std::initializer_list<size_t> const& new_shape) {
      return subview_impl(start, new_shape);
    }

    /**
     * @brief Returns whether the tensor is a view of another tensor
     *
     */
    bool isSubView() const { return size_ != storage_ptr_->size(); }

    size_t calcSize() const { return product(shape_); }

   private:
    /**
    * Constructor that accepts a pointer to existing TensorStorage, together
    * with shape and offset.
    * @param storage pointer to TensorStorage
    * @param offset offset from first element of storage
    * @param shape shape of the Tensor
    * @return
    */
    explicit Tensor(const TensorStoragePointer storage,
                    const size_t offset,
                    std::initializer_list<size_t> const& shape) {
      offset_      = offset;
      size_        = product(shape);
      storage_ptr_ = storage;
      std::copy(shape.begin(), shape.end(), shape_.begin());
    }

    explicit Tensor(const TensorStoragePointer storage,
                    const size_t offset,
                    std::array<size_t, kDimensions> const& shape) {
      offset_      = offset;
      size_        = product(shape);
      storage_ptr_ = storage;
      std::copy(shape.begin(), shape.end(), shape_.begin());
    }

    /*
     * Implementation method to extract a view from a tensor
     * Raises an exception when sizes of the starting offset and new_shape
     * are bigger than the current dimensions number. Also raises an exception
     * when the requested view size is not feasible.
     */
    Tensor subview_impl(std::initializer_list<size_t> const& start,
                        std::initializer_list<size_t> const& new_shape) {
      if (start.size() > kDimensions || new_shape.size() > kDimensions) {
        throw simple_error("Overpassed number of existing dimensions.");
      }

      // compute the new offset and check that it's feasible to create
      // the new view.
      const size_t new_offset = offset_ + compute_offset(start, shape_);
      if (new_offset + product(new_shape) > size_) {
        throw simple_error("Cannot create a view from this tensor");
      }

      return Tensor(storage_ptr_, new_offset, new_shape);
    }

    /**
     * A tensor holds data in C-style nD array, i.e row-major order:
     * the rightmost index ?varies the fastest?.
     */
    std::array<size_t, kDimensions> shape_;

    /* Offset from the beginning of TensorStorage */
    size_t offset_;
    size_t size_;

    /* pointer to TensorStorage */
    TensorStoragePointer storage_ptr_;
  };
}  // namespace simpleCNN

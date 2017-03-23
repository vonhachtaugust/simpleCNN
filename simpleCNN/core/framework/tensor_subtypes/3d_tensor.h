//
// Created by hacht on 3/13/17.
//
#pragma once

#include "../tensor.h"

namespace simpleCNN {
  /*

  /**
  * In case of three dimensional tensor, use these
  * when fetching height / width / depth values. Also
  * found using shape, but kept like this for keep track
  * of the common definition.
  */
  // enum dimension_t { height = 0, width = 1, depth = 2 };
  // enum dimension_t { depth = 0, height = 1, width = 2 };

  /**
   * @breif Tensor subclass implementation to handle the
   * many calls to width / height / depth dependent operations.
   * Also allows for easier handling of which component type is
   * currently in operation.
   *
   */
  template <typename T = float_t, bool kConst = false, typename Allocator = aligned_allocator<T, 64>>
  class Tensor_3 : public Tensor<T, 3, kConst, Allocator> {
   public:
    typedef Tensor<T, 3, kConst, Allocator> Base;

    Tensor_3() { component_ = component_t::UNSPECIFIED; }

    explicit Tensor_3(const std::initializer_list<size_t>& shape) : Base(shape) {
      component_ = component_t::UNSPECIFIED;
    }

    explicit Tensor_3(component_t component) : Base() { component_ = component; }

    explicit Tensor_3(const std::initializer_list<size_t>& shape, component_t component) : Base(shape) {
      component_ = component;
    }

    /**
    * @return type of component this type constitutes
    */
    component_t get_component_type() const { return component_; }

    /**
     * @param comp = composition type
     * @return itself, for making multiple 'set' function calls at once
     */
    Tensor_3& set_component_type(component_t comp) {
      // component_ = comp;
      return *this;
    }

    // inline size_t height() const { return this->shape()[dimension_t::height]; }

    // inline size_t width() const { return this->shape()[dimension_t::width]; }

    // inline size_t depth() const { return this->shape()[dimension_t::depth]; }

   private:
    /**
     * Name used to distiquish between data, weight, bias etc.
     **/
    component_t component_;
  };
}  // namespace simpleCNN

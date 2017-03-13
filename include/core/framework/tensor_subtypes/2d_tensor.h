//
// Created by hacht on 3/13/17.
//
#pragma once

#include "../tensor.h"

namespace simpleCNN {
    /**
    * In case of two dimensional tensor, use these
    * when fetching row / cols values. Also
    * found using shape, but kept like this for keep track
    * of the common definition.
    */
    enum matrix_dimension_t {
        row = 0,
        col = 1
    };

    /**
     * @breif Tensor subclass implementation to model a matrix.
     * Allows for easier handling of matrix operations.
     *
     */
    template<typename T = float_t,
            bool kConst = false,
            typename Allocator = aligned_allocator<T, 64>>
    class Tensor_2 : public Tensor<T, 2, kConst, Allocator> {
    public:
        typedef Tensor<T, 2, kConst, Allocator> Base;

        explicit Tensor_2(const std::initializer_list<size_t>& shape)
                :Base(shape) { }

        inline size_t rows() const { return this->shape()[matrix_dimension_t::row]; }

        inline size_t cols() const { return this->shape()[matrix_dimension_t::col]; }

    private:
    };
} // namespace simpleCNN

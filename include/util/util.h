//
// Created by hacht on 3/4/17.
//

#pragma once

#include "../core/framework/tensor_subtypes/3d_tensor.h"
#include "../core/framework/tensor_subtypes/2d_tensor.h"

namespace simpleCNN
{
    template<typename T = float_t, typename Allocator = aligned_allocator<T, 64>>
    using default_vec_t = std::vector<T, Allocator>;
    using vec_t = default_vec_t<>;

    /* ------------------------------------------------------------------- //
     * Matrix type used in matrix multiplications
     */
    template <typename T = float_t,
              bool kConst = false,
              typename Allocator = aligned_allocator<T, 64>>
    using default_matrix_t = Tensor_2<T, kConst, Allocator>;
    using matrix_t = default_matrix_t<>; // no more <> yey
    using matrix_ptr_t = std::shared_ptr<matrix_t>;
    using matrix_data_t = std::vector<matrix_t>;
    using vec_matrix_ptr_t = std::vector<matrix_ptr_t>;

    /* ------------------------------------------------------------------- //
     * Use vector<tensor_t> instead of tensor_t<4>
     * The shape difference of weights, bias, data etc makes
     * it difficult to store all in a 4-d tensor unless one
     * wants to be a subview ninja.
     */
    template <typename T = float_t,
              bool kConst = false,
              typename Allocator = aligned_allocator<T, 64>>
    using default_tensor_t = Tensor_3<T, kConst, Allocator>;

    using tensor_t = default_tensor_t<>; // removed <>  yey
    using tensor_ptr_t = std::shared_ptr<tensor_t>;
    using vec_tensor_ptr_t = std::vector<tensor_ptr_t>;

    using data_t = std::vector<tensor_t>;
    using data_ptr_t = std::shared_ptr<tensor_t>;
    // ------------------------------------------------------------------- //

    /* ------------------------------------------------------------------- //
     * Constructor arguments require data type composition for initialization
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
} // namespace simpleCNN

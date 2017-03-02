//
// Created by hacht on 3/1/17.
//

#pragma once

#include <term.h>
#include <cmath>
#include <cstdlib>
#include "../core/framework/tensor.h"

namespace simpleCNN
{
    template<typename T     = float_t,
             size_t K       = size_t(0),
             size_t F       = size_t(0),
             size_t S       = size_t(1),
             size_t P       = size_t(0)>
    class Conv_layer{

    public:
    Conv_layer() {};


        /*
         * Number of filters
         */
        const size_t & num_filters() const { return K; }

        /*
         * The filters spatial extent
         */
        const size_t & spatial_extend() const { return F; }

        /*
         * The stride
         */
        const size_t & stride() const { return S; }

        /*
         * The padding
         */
        const size_t & padding() const { return P; }

    private:
        Tensor<T, 3, false> filters;
    };
}
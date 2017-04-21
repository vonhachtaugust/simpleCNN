//
// Created by hacht on 4/3/17.
//

#pragma once

#include "../../util/util.h"
#include "params.h"

namespace simpleCNN {
  namespace core {

    class Maxpooling_params : public Params {
     public:
      // Input parameters
      size_t input_width;
      size_t input_height;
      size_t in_channels;
      size_t batch_size;

      // filter parameters
      size_t pooling_size_x;
      size_t pooling_size_y;
      size_t stride_x;
      size_t stride_y;

      // Output parameters
      size_t output_width;
      size_t output_height;
      size_t out_channels;

      // Keep track of winning tile in forward pass
      tensor_t max_index;

      const Maxpooling_params &maxpool() const { return *this; }

      size_t pooling_size() const {
        if (pooling_size_x == pooling_size_y) {
          return pooling_size_x;
        }
        std::cout << pooling_size_x << "\t" << pooling_size_y << std::endl;
        throw simple_error(
          "Error: Filter sizes are different, therefore filter size is "
          "undefined");
      }

      size_t stride() const {
        if (stride_x == stride_y) {
          return stride_x;
        }
        std::cout << stride_x << "\t" << stride_y << std::endl;
        throw simple_error("Error: Stride sizes are different, therefore stride is undefined");
      }

     private:
    };
  }  // namespace core
}  // namespace simpleCNN

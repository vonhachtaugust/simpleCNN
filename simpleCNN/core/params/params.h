//
// Created by hacht on 3/4/17.
//

#pragma once

namespace simpleCNN {
  namespace core {
    class Params {
     public:
      Params() {}

      inline size_t conv_out_length(size_t image_side_length,
                                    size_t filter_side_lenght,
                                    size_t stride,
                                    size_t padding) const {
        return (image_side_length - filter_side_lenght + 2 * padding) / stride + 1;
      }
    };
  }  // namespace core
}  // namespace simpleCNN

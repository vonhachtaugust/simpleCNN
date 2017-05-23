//  // Created by hacht on 3/8/17.  //
#pragma once

#include "../../util/util.h"
#include "params.h"

#ifdef USE_CUDNN
#include "../../../third_party/cudnn/include/cudnn.h"
#include "../../util/cuda_utils.h"
#endif

namespace simpleCNN {
  namespace core { /**
         * @breif Convolution settings
         *
         * currently only supports 2D
                      convolution by s(tr/l)iding
         * window for each
                      depth/channel.
         */
    class Conv_params : public Params {
     public:
      // Input parameters
      size_t input_width;
      size_t input_height;
      size_t in_channels;
      size_t batch_size;

      // Filter parameters (num filters = out_channels)
      size_t filter_width;
      size_t filter_height;
      size_t horizontal_stride;
      size_t vertical_stride;
      size_t padding;
      bool has_bias;

      // Output parameters
      size_t output_width;
      size_t output_height;
      size_t out_channels;

      const Conv_params& conv() const {
        return *this;
      } /**
               * @breif common case of symmetric striding
               *
           */
      size_t stride() const {
        if (vertical_stride == horizontal_stride) {
          return vertical_stride;
        }
        std::cout << vertical_stride << "\t" << horizontal_stride << std::endl;
        throw simple_error("Error: Stride sizes are different, stride is undefined");
      } /**
               * @brief common case that filter size is symmetric
           *
               */
      size_t filter_size() const {
        if (filter_width == filter_height) {
          return filter_width;
        }
        std::cout << filter_width << "\t" << filter_height << std::endl;
        throw simple_error("Error: Filter sizes are different, therefore filter size is undefined");
      }
    };
  }  // namespace core
}  // namespace simpleCNN

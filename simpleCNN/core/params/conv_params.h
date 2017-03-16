//  // Created by hacht on 3/8/17.  //
#pragma once
#include "../../util/util.h"
#include "params.h"
namespace simpleCNN {
  namespace core { /**
         * @breif Convolution settings (in an messy way
                      ...)
         *
         * currently only supports 2D
                      convolution by s(tr/l)iding
         * window for each
                      depth/channel.
         */
    class Conv_params : public Params {
     public:
      tensor_t in;
      tensor_t out;
      tensor_t weights;
      bool has_bias;
      size_t padding;
      size_t vertical_stride;
      size_t horizontal_stride;
      inline Conv_params conv() {
        return *this;
      } /**
               * @breif common case of symmetric striding
               *
           */
      inline size_t stride() const {
        if (vertical_stride == horizontal_stride) {
          return vertical_stride;
        }
        throw simple_error(
          "Error: Stride sizes are different, stride is undefined");
      } /**
               * @brief common case that filter size is symmetric
           *
               */
      inline size_t filter_size() const {
        if (weights.height() == weights.width()) {
          return weights.height();
        }
        throw simple_error(
          "Error: Filter sizes are different, filter size is undefined");
      } /**
               * @breif equivalent definition in computer vision terms
           *
               */
      inline size_t channels() const {
        return weights.depth();
      } /**
           ----------------------------------------------------------------------
           //
               * @breif Number of filter locations of which the
           convolutional filters
               * will be applied. Depends on image
           size, filter size, stride and padding
               * according to:
           (image_size + 2*padding - filter_size) / stride + 1
               * */
      inline size_t conv_out_length(size_t image_side_length,
                                    size_t filter_side_length,
                                    size_t stride,
                                    size_t padding) const {
        return (image_side_length - filter_side_length + 2 * padding) / stride +
               1;
      }

     private:
    };
  }  // namespace core
}  // namespace simpleCNN
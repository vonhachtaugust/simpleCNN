//
// Created by hacht on 3/2/17.
//

#pragma once

#include <stdio.h>
#include "../core/framework/tensor.h"

namespace simpleCNN {

  template <typename T = float_t>
  T im2col_get_pixel(const tensor_t& image,
                     int image_width,
                     int image_height,
                     int depth,
                     int height,
                     int width,
                     int pad) {
    height -= pad;
    width -= pad;

    if (height < 0 || width < 0 || height >= image_height ||
        width >= image_width) {
      return T(0);
    }
    return image.host_at(height, width, depth);
  }

  /**
   * The core idea is to turn the shape of the data (previously 3-dim tensor)
   *into
   *a matrix which
   * allows for efficient matrix multiplication:
   * For each position at which a filter will be applied to in the image, these
   *values at those positions
   * are stretched out into a column. So the end product X_col is a matrix with
   *a
   *column for each filter
   * representing the data onto which a filter would be applied onto, this later
   *will be done via matrix
   * multiplication with a similar stretching of the filters, but put into rows.
   *
   * Default values are for converting image into col as is i.e. without
   *modification.
   *
   **/
  template <typename T = float_t>
  void im2col_cpu(const tensor_t& image,
                  matrix_t& X_col,
                  int channels,
                  int image_height,
                  int image_width,
                  int filter_size = 1,
                  int stride      = 1,
                  int pad         = 0) {
    int c, h, w;
    int height_col = (image_height + 2 * pad - filter_size) / stride + 1;
    int width_col  = (image_width + 2 * pad - filter_size) / stride + 1;

    int X_col_num_rows = channels * filter_size * filter_size;
    for (c = 0; c < X_col_num_rows; ++c) {
      int image_width_offset  = c % filter_size;  // fastest index
      int image_height_offset = (c / filter_size) % filter_size;
      int image_channel = (c / filter_size / filter_size) /* % filter_size */;
      for (h = 0; h < height_col; ++h) {
        int image_row = image_height_offset + h * stride;
        for (w = 0; w < width_col; ++w) {
          int image_col = image_width_offset + w * stride;
          int col_index = w + width_col * h;
          X_col.host_at(c, col_index) =
            im2col_get_pixel(image, image_width, image_height, image_channel,
                             image_row, image_col, pad);
        }
      }
    }
  }

  template <typename T = float_t>
  void col2im_add_pixel(const matrix_t& result,
                        tensor_t& image,
                        int image_width,
                        int image_height,
                        int depth,
                        int height,
                        int width,
                        int pad,
                        T val) {
    height -= pad;
    width -= pad;

    if (height < 0 || width < 0 || height >= image_height ||
        width >= image_width) {
      return;
    }
    image.host_at(height, width, depth) += val;
  }

  /*
   * Default values are for converting col to image as is, i.e. without
   * modification
   *
   */
  template <typename T = float_t>
  void col2im_cpu(const matrix_t& result,
                  tensor_t& image,
                  int channels,
                  int image_height,
                  int image_width,
                  int filter_size = 1,
                  int stride      = 1,
                  int pad         = 0) {
    int c, h, w;
    int height_col = (image_height + 2 * pad - filter_size) / stride + 1;
    int width_col  = (image_width + 2 * pad - filter_size) / stride + 1;

    int result_num_rows = channels * filter_size * filter_size;
    for (c = 0; c < result_num_rows; ++c) {
      int image_width_offset  = c % filter_size;  // fastest index
      int image_height_offset = (c / filter_size) % filter_size;
      int image_channel = (c / filter_size / filter_size) /* % filter_size */;
      for (h = 0; h < height_col; ++h) {
        int image_row = image_height_offset + h * stride;
        for (w = 0; w < width_col; ++w) {
          int image_col = image_width_offset + w * stride;
          int col_index = w + width_col * h;
          T val         = result.host_at(c, col_index);
          col2im_add_pixel(result, image, image_width, image_height,
                           image_channel, image_row, image_col, pad, val);
        }
      }
    }
  }
}  // namespace simpleCNN

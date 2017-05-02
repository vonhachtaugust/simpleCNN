//
// Created by hacht on 3/2/17.
//

#pragma once

#include <stdio.h>
#include "../core/framework/tensor.h"

namespace simpleCNN {

  template <typename T = float_t>
  T im2col_get_pixel(const tensor_t& image,
                     int batch_number,
                     int image_width,
                     int image_height,
                     int depth,
                     int height,
                     int width,
                     int pad) {
    height -= pad;
    width -= pad;

    // std::cout << "[" << batch_number << ", " <<  depth << ", " << height << ", " << width << "]" << std::endl;

    if (height < 0 || width < 0 || height >= image_height || width >= image_width) {
      return T(0);
    }
    return image.host_at(batch_number, depth, height, width);
  }

  /**
   * The core idea is to turn the shape of the data (previously 4-dim tensor)
   *into
   *a matrix which
   * allows for efficient matrix multiplication:
   * For each position at which a filter will be applied to in the image, these
   *values at those positions
   * are stretched out into a column. So the end product output is a matrix with
   *a
   *column/row for each filter
   * representing the data onto which a filter would be applied onto, this later
   *will be done via matrix
   * multiplication with a similar stretching of the filters, but put into rows.
   *
   * Default values are for converting image into col as is i.e. without
   *modification.
   *
   * @param image               [in] 3D tensor to convert from
    * @param output             [in] matrix to convert to
    * @param channels           [in] Number of WEIGHT channels
    * @param image_height       [in] Single channel (2D) image height without padding
    * @param image_width        [in] Single channel (2D) image width without padding
    * @param filter_size        [in] Size of each filter (weight channel)
    * @param stride             [in]
    * @param padding            [in]
    * @param                    [in]
   *
   **/
  template <typename T = float_t>
  void im2col_cpu(const tensor_t& image,
                  int batch_number,
                  matrix_t& output,
                  int channels,
                  int image_height,
                  int image_width,
                  int filter_size = 1,
                  int stride      = 1,
                  int pad         = 0) {
    int c, h, w;
    int height_col = (image_height + 2 * pad - filter_size) / stride + 1;
    int width_col  = (image_width + 2 * pad - filter_size) / stride + 1;

    // image data has to be stretched into activate column of size equal to the
    // size of the weights
    int output_num_rows = channels * filter_size * filter_size;
    for (c = 0; c < output_num_rows; ++c) {
      int image_width_offset  = c % filter_size;  // fastest index
      int image_height_offset = (c / filter_size) % filter_size;
      int image_channel       = (c / filter_size / filter_size) /* % filter_size */;
      for (h = 0; h < height_col; ++h) {
        int image_row = image_height_offset + h * stride;
        for (w = 0; w < width_col; ++w) {
          int image_col = image_width_offset + w * stride;
          int col_index = w + width_col * h;
          output.host_at(c, col_index) =
            im2col_get_pixel(image, batch_number, image_width, image_height, image_channel, image_row, image_col, pad);
        }
      }
    }
  }

  template <typename T = float_t>
  void im2row_flipped_cpu(const tensor_t& image,
                          matrix_t& output,
                          int out_channels,
                          int channels,
                          int image_height,
                          int image_width,
                          int filter_size = 1,
                          int stride      = 1,
                          int pad         = 0) {
    int c, h, w, oc;
    int height_col       = (image_height + 2 * pad - filter_size) / stride + 1;
    int width_col        = (image_width + 2 * pad - filter_size) / stride + 1;
    int max_width_index  = width_col - 1;
    int max_height_index = height_col - 1;

    // image data has to be stretched into activate column of size equal to the
    // size of the weights
    for (oc = 0; oc < out_channels; ++oc) {
      int output_num_rows = channels * filter_size * filter_size;
      for (c = 0; c < output_num_rows; ++c) {
        int image_width_offset  = c % filter_size;  // fastest index
        int image_height_offset = (c / filter_size) % filter_size;
        int image_channel       = (c / filter_size / filter_size) /* % filter_size */;
        for (h = 0; h < height_col; ++h) {
          int image_row = image_height_offset + h * stride;
          for (w = 0; w < width_col; ++w) {
            int image_col = image_width_offset + w * stride;
            int col_index = (oc * height_col + h) * width_col + w;
            output.host_at(c, col_index) =
              im2col_get_pixel(image, oc, image_width, image_height, image_channel, max_height_index - image_row,
                               max_width_index - image_col, pad);
          }
        }
      }
    }
  }

  template <typename T = float_t>
  void im2row_cpu(const tensor_t& image,
                  matrix_t& output,
                  int out_channels,
                  int channels,
                  int image_height,
                  int image_width,
                  int filter_size = 1,
                  int stride      = 1,
                  int pad         = 0) {
    int c, h, w, oc;
    int height_col = (image_height + 2 * pad - filter_size) / stride + 1;
    int width_col  = (image_width + 2 * pad - filter_size) / stride + 1;

    // image data has to be stretched into activate column of size equal to the
    // size of the weights
    for (oc = 0; oc < out_channels; ++oc) {
      int output_num_rows = channels * filter_size * filter_size;
      for (c = 0; c < output_num_rows; ++c) {
        int image_width_offset  = c % filter_size;  // fastest index
        int image_height_offset = (c / filter_size) % filter_size;
        int image_channel       = (c / filter_size / filter_size) /* % filter_size */;
        for (h = 0; h < height_col; ++h) {
          int image_row = image_height_offset + h * stride;
          for (w = 0; w < width_col; ++w) {
            int image_col = image_width_offset + w * stride;
            int col_index = (c * height_col + h) * width_col + w;
            output.host_at(oc, col_index) =
              im2col_get_pixel(image, oc, image_width, image_height, image_channel, image_row, image_col, pad);
          }
        }
      }
    }
  }

  template <typename T = float_t>
  void col2im_add_pixel(const matrix_t& result,
                        int batch_number,
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

    if (height < 0 || width < 0 || height >= image_height || width >= image_width) {
      return;
    }
    image.host_at(batch_number, depth, height, width) += val;
  }

template <typename T = float_t>
void col2im_insert_pixel(const matrix_t& result,
                      int batch_number,
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

  if (height < 0 || width < 0 || height >= image_height || width >= image_width) {
    return;
  }
  image.host_at(batch_number, depth, height, width) = val;
}

  /*
   * Default values are for converting col to image as is, i.e. without
   * modification
   *
   */
  template <typename T = float_t>
  void col2im_insert_cpu(const matrix_t& result,
                  int batch_number,
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
      int image_channel       = (c / filter_size / filter_size) /* % filter_size */;
      for (h = 0; h < height_col; ++h) {
        int image_row = image_height_offset + h * stride;
        for (w = 0; w < width_col; ++w) {
          int image_col = image_width_offset + w * stride;
          int col_index = w + width_col * h;
          T val         = result.host_at(c, col_index);
          col2im_insert_pixel(result, batch_number, image, image_width, image_height, image_channel, image_row, image_col,
                           pad, val);
        }
      }
    }
  }

/*
 * Default values are for converting col to image as is, i.e. without
 * modification
 *
 */
template <typename T = float_t>
void col2im_add_cpu(const matrix_t& result,
                       int batch_number,
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
    int image_channel       = (c / filter_size / filter_size) /* % filter_size */;
    for (h = 0; h < height_col; ++h) {
      int image_row = image_height_offset + h * stride;
      for (w = 0; w < width_col; ++w) {
        int image_col = image_width_offset + w * stride;
        int col_index = w + width_col * h;
        T val         = result.host_at(c, col_index);
        col2im_add_pixel(result, batch_number, image, image_width, image_height, image_channel, image_row, image_col,
                            pad, val);
      }
    }
  }
}

  template <typename T = float_t>
  void row2im_add_cpu(const matrix_t& result,
                        tensor_t& image,
                        int out_channels,
                        int channels,
                        int image_height,
                        int image_width,
                        int filter_size = 1,
                        int stride      = 1,
                        int pad         = 0) {
    int c, h, w, oc;
    int height_col = (image_height + 2 * pad - filter_size) / stride + 1;
    int width_col  = (image_width + 2 * pad - filter_size) / stride + 1;

    for (oc = 0; oc < out_channels; ++oc) {
      int result_num_rows = channels * filter_size * filter_size;
      for (c = 0; c < result_num_rows; ++c) {
        int image_width_offset  = c % filter_size;  // fastest index
        int image_height_offset = (c / filter_size) % filter_size;
        int image_channel       = (c / filter_size / filter_size) /* % filter_size */;
        for (h = 0; h < height_col; ++h) {
          int image_row = image_height_offset + h * stride;
          for (w = 0; w < width_col; ++w) {
            int image_col = image_width_offset + w * stride;
            int col_index = (c * height_col + h) * width_col + w;
            T val         = result.host_at(col_index, oc);
            col2im_add_pixel(result, oc, image, image_width, image_height, image_channel, image_row, image_col, pad,
                             val);
          }
        }
      }
    }
  }

  template <typename T = float_t>
  void row2im_non_added_cpu(const matrix_t& result,
                            tensor_t& image,
                            int batch_index,
                            int out_channels,
                            int channels,
                            int image_height,
                            int image_width,
                            int filter_size = 1,
                            int stride      = 1,
                            int pad         = 0) {
    int c, h, w, oc;
    int height_col = (image_height + 2 * pad - filter_size) / stride + 1;
    int width_col  = (image_width + 2 * pad - filter_size) / stride + 1;

    for (oc = 0; oc < out_channels; ++oc) {
      int result_num_rows = channels * filter_size * filter_size;
      for (c = 0; c < result_num_rows; ++c) {
        int image_width_offset  = c % filter_size;  // fastest index
        int image_height_offset = (c / filter_size) % filter_size;
        int image_channel       = (c / filter_size / filter_size) /* % filter_size */;
        for (h = 0; h < height_col; ++h) {
          int image_row = image_height_offset + h * stride;
          for (w = 0; w < width_col; ++w) {
            int image_col = image_width_offset + w * stride;
            int col_index = (c * height_col + h) * width_col + w;
            T val         = result.host_at(col_index, oc);
            col2im_add_pixel(result, batch_index, image, image_width, image_height, image_channel, image_row, image_col,
                             pad, val);
          }
        }
      }
    }
  }
}  // namespace simpleCNN

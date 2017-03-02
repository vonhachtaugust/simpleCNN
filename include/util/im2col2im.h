//
// Created by hacht on 3/2/17.
//
#pragma once

#include <stdio.h>
#include "../core/framework/tensor.h"

namespace simpleCNN {

    template<typename T = float_t>
    T im2col_get_pixel(
            const Tensor<T, 3, false>& image,
            int image_width,
            int image_height,
            int channel,
            int row,
            int col,
            int pad
    ) {
        row -= pad;
        col -= pad;

        if (row < 0 || col < 0 || row >= image_height || col >= image_width)
        {
            return T(0);
        }
        return image.host_at(channel, row, col);
    }

    /**
     * The core idea is to turn the shape of the data (previously 3-dim tensor) into a matrix which
     * allows for efficient matrix multiplication:
     * For each position at which a filter will be applied to in the image, these values at those positions
     * are stretched out into a column. So the end product X_col is a matrix with a column for each filter
     * representing the data onto which a filter would be applied onto, this later will be done via matrix
     * multiplication with a similar stretching of the filters.
     *
     **/
    template<typename T = float_t>
    void im2col_cpu(
            const Tensor<T, 3, false>& image,
            Tensor<T, 2, false>& X_col,
            int channels,
            int image_height,
            int image_width,
            int filter_size,
            int stride,
            int pad) {

        int c, h, w;
        int height_col = (image_height + 2 * pad - filter_size)/stride + 1;
        int width_col = (image_width + 2 * pad - filter_size)/stride + 1;

        int X_col_num_rows = channels * filter_size * filter_size;
        for (c = 0; c < X_col_num_rows; ++c)
        {
            int image_width_offset = c % filter_size; // fastest index
            int image_height_offset = (c / filter_size) % filter_size;
            int image_channel = (c / filter_size / filter_size) /* % filter_size */;
            for (h = 0; h < height_col; ++h)
            {
                int image_row = image_height_offset + h * stride;
                for (w = 0; w < width_col; ++w)
                {
                    int image_col = image_width_offset + w * stride;
                    int col_index = w + width_col * h;
                    X_col.host_at(c, col_index) = im2col_get_pixel(image, image_width, image_height, image_channel, image_row, image_col, pad);

                    // int col_index = (c * height_col + h) * width_col + w;
                    //data_col[col_index] = im2col_get_pixel(data_im, height, width, channels,
                    //        im_row, im_col, c_im, pad);
                }
            }
        }
    }

    template<typename T = float_t>
    void col2im_add_pixel(
            const Tensor<T, 2, false>& result,
            Tensor<T, 3, false>& image,
            int image_width,
            int image_height,
            int row,
            int col,
            int channel,
            int pad,
            T val
    ) {
        row -= pad;
        col -= pad;

        if (row < 0 || col < 0 || row >= image_height || col >= image_width)
         {
            return;
         }
        image.host_at(channel, row, col) += val;
    }

    /*
     * Default values are for arbitrary col to image convertion. Other values require specific
     * shapes of result and image (i.e. not recommended).
     */
    template<typename T = float_t>
    void col2im_cpu(
            const Tensor<T, 2>& result,
            Tensor<T, 3>& image,
            int channels,
            int image_height,
            int image_width,
            int filter_size = 1,
            int stride      = 1,
            int pad         = 0) {

        int c, h, w;
        int height_col = (image_height + 2 * pad - filter_size)/stride + 1;
        int width_col = (image_width + 2 * pad - filter_size)/stride + 1;

        int result_num_rows = channels * filter_size * filter_size;
        for (c = 0; c < result_num_rows; ++c)
        {
            int image_width_offset = c % filter_size; // fastest index
            int image_height_offset = (c / filter_size) % filter_size;
            int image_channel = (c / filter_size / filter_size) /* % filter_size */;
            for (h = 0; h < height_col; ++h)
            {
                int image_row = image_height_offset + h * stride;
                for (w = 0; w < width_col; ++w)
                {
                    int image_col = image_width_offset + w * stride;
                    int col_index = w + width_col * h;
                    T val = result.host_at(c, col_index);
                    col2im_add_pixel(result, image, image_width, image_height, image_row, image_col, image_channel, pad, val);
                    //result.host_at(c, col_index) = im2col_get_pixel(image, image_width, image_height, image_channel, image_row, image_col, pad);

                    // int col_index = (c * height_col + h) * width_col + w;
                    // double val = data_col[col_index];
                    // col2im_add_pixel(data_im, height, width, channels,
                    //        im_row, im_col, c_im, pad, val);
                }
            }
        }
    }

}
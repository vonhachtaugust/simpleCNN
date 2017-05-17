//
// Created by hacht on 4/25/17.
//

#pragma once

#include <fstream>
#include "byteswap.h"

namespace simpleCNN {
  namespace parser_details {

    struct mnist_image_header {
      uint32_t magic_number;
      uint32_t num_images;
      uint32_t num_rows;
      uint32_t num_cols;
    };

    struct mnist_label_header {
      uint32_t magic_number;
      uint32_t num_items;
    };

    inline void parse_mnist_header(std::ifstream &ifs, mnist_image_header &header) {
      ifs.read(reinterpret_cast<char *>(&header.magic_number), 4);
      ifs.read(reinterpret_cast<char *>(&header.num_images), 4);
      ifs.read(reinterpret_cast<char *>(&header.num_rows), 4);
      ifs.read(reinterpret_cast<char *>(&header.num_cols), 4);

      if (!is_high_endian()) {
        header.magic_number = swap_uint32(header.magic_number);
        header.num_images   = swap_uint32(header.num_images);
        header.num_rows     = swap_uint32(header.num_rows);
        header.num_cols     = swap_uint32(header.num_cols);
      }

      if (header.magic_number != 0x00000803 || header.num_images <= 0) {
        throw simple_error("File format error.");
      }
      if (ifs.bad() || ifs.fail()) {
        throw simple_error("File error.");
      }
    }

    inline void parse_mnist_header(std::ifstream &ifs, mnist_label_header &header) {
      ifs.read(reinterpret_cast<char *>(&header.magic_number), 4);
      ifs.read(reinterpret_cast<char *>(&header.num_items), 4);

      if (!is_high_endian()) {
        header.magic_number = swap_uint32(header.magic_number);
        header.num_items    = swap_uint32(header.num_items);
      }

      if (header.magic_number != 0x00000801 || header.num_items <= 0) {
        throw simple_error("File format error.");
      }

      if (ifs.bad() || ifs.fail()) {
        throw simple_error("File error.");
      }
    }

    inline void parse_mnist_image(std::ifstream &ifs,
                                  const mnist_image_header &header,
                                  float_t scale_min,
                                  float_t scale_max,
                                  int x_padding,
                                  int y_padding,
                                  size_t start_index,
                                  tensor_t *dst) {
      const int width = header.num_cols + 2 * x_padding;
      // const int height = header.num_rows + 2 * y_padding;

      std::vector<uint8_t> vec(header.num_cols * header.num_rows);

      ifs.read(reinterpret_cast<char *>(&vec[0]), header.num_cols * header.num_rows);

      for (uint32_t y = 0; y < header.num_rows; ++y) {
        for (uint32_t x = 0; x < header.num_cols; ++x) {
          auto index = x + x_padding + width * (y + y_padding);
          dst->host_at_index(start_index + index) =
            (vec[y * header.num_cols + x] / float_t(255)) * (scale_max - scale_min) + scale_min;
        }
      }
    }
  }  // namespace parser_details

  inline void parse_mnist_labels(const std::string &file, tensor_t *labels, const size_t subset = 1) {
    std::ifstream ifs(file.c_str(), std::ios::in | std::ios::binary);

    if (ifs.bad() || ifs.fail()) {
      throw simple_error("Failed to open file: " + file);
    }

    parser_details::mnist_label_header header;
    parser_details::parse_mnist_header(ifs, header);

    // labels->resize(header.num_items);
    for (uint32_t i = 0; i < (header.num_items / subset); ++i) {
      uint8_t label;
      ifs.read(reinterpret_cast<char *>(&label), 1);
      labels->host_at_index(i) = static_cast<float_t>(label);
    }
  }

  inline void parse_mnist_images(const std::string &file,
                                 tensor_t *images,
                                 float_t scale_min,
                                 float_t scale_max,
                                 int x_padding,
                                 int y_padding,
                                 const size_t subset = 1) {
    if (x_padding < 0 || y_padding < 0) {
      throw simple_error("Non-negative padding size required.");
    }
    if (scale_min >= scale_max) {
      throw simple_error("Max scale must be larger than min scale.");
    }

    std::ifstream ifs(file.c_str(), std::ios::in | std::ios::binary);

    if (ifs.bad() || ifs.fail()) {
      throw simple_error("Failed to open file: " + file);
    }

    parser_details::mnist_image_header header;
    parser_details::parse_mnist_header(ifs, header);

    size_t n = images->size() / images->shape()[0];
    for (uint32_t i = 0; i < (header.num_images / subset); ++i) {
      parser_details::parse_mnist_image(ifs, header, scale_min, scale_max, x_padding, y_padding, i * n, images);
    }
  }
}  // namespace simpleCNN

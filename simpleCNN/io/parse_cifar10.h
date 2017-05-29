//
// Created by hacht on 5/29/17.
//

#pragma once

#pragma once
#include <algorithm>
#include <cstdint>
#include <fstream>
#include "../util/util.h"

#define CIFAR10_IMAGE_DEPTH (3)
#define CIFAR10_IMAGE_WIDTH (32)
#define CIFAR10_IMAGE_HEIGHT (32)
#define CIFAR10_IMAGE_AREA (CIFAR10_IMAGE_WIDTH * CIFAR10_IMAGE_HEIGHT)
#define CIFAR10_IMAGE_SIZE (CIFAR10_IMAGE_AREA * CIFAR10_IMAGE_DEPTH)

namespace simpleCNN {

  /**
   * parse CIFAR-10 database format images
   *
   * @param filename [in] filename of database(binary version)
   * @param train_images [out] parsed images
   * @param train_labels [out] parsed labels
   * @param scale_min  [in]  min-value of output
   * @param scale_max  [in]  max-value of output
   * @param x_padding  [in]  adding border width (left,right)
   * @param y_padding  [in]  adding border width (top,bottom)
   **/
  inline void parse_cifar10(const std::string &filename,
                            tensor_t *train_images,
                            tensor_t *train_labels,
                            float_t scale_min,
                            float_t scale_max,
                            int x_padding,
                            int y_padding,
                            const size_t start_image = 0,
                            const size_t start_label = 0,
                            const size_t subset      = 1) {
    if (x_padding < 0 || y_padding < 0) throw simple_error("padding size must not be negative");
    if (scale_min >= scale_max) throw simple_error("scale_max must be greater than scale_min");

    std::ifstream ifs(filename.c_str(), std::ios::in | std::ios::binary);
    if (ifs.fail() || ifs.bad()) throw simple_error("failed to open file:" + filename);

    uint8_t label;
    size_t start_index = start_image;
    size_t counter     = start_label;
    std::vector<unsigned char> buf(CIFAR10_IMAGE_SIZE);

    while (ifs.read(reinterpret_cast<char *>(&label), 1)) {
      if (!ifs.read(reinterpret_cast<char *>(&buf[0]), CIFAR10_IMAGE_SIZE)) break;
      int w = CIFAR10_IMAGE_WIDTH;
      int h = CIFAR10_IMAGE_HEIGHT;

      for (int c = 0; c < CIFAR10_IMAGE_DEPTH; c++) {
        for (int y = 0; y < CIFAR10_IMAGE_HEIGHT; y++) {
          for (int x = 0; x < CIFAR10_IMAGE_WIDTH; x++) {
            auto index = c * w * h + (y + y_padding) * w + x + x_padding;
            auto val =
              scale_min +
              (scale_max - scale_min) * buf[c * CIFAR10_IMAGE_AREA + y * CIFAR10_IMAGE_WIDTH + x] / float_t(255);
            train_images->host_at_index(start_index + index) = val;
          }
        }
      }

      train_labels->host_at_index(counter) = label;
      start_index += CIFAR10_IMAGE_SIZE;
      counter++;
    }
  }

}  // namespace simpleCNN

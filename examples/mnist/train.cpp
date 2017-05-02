//
// Created by hacht on 4/25/17.
//

#include <iostream>
#include "../../simpleCNN/simpleCNN.h"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace simpleCNN;
using namespace cv;

using dropout = Dropout_layer;
using conv    = Convolutional_layer<float_t, activation::ReLU<float_t>>;
using conv_I  = Convolutional_layer<float_t, activation::Identity<float_t>>;
using maxpool = Maxpooling_layer<>;
using fully   = Connected_layer<>;
using classy  = Connected_layer<float_t, activation::Softmax<float_t>>;
using class_I = Connected_layer<float_t, activation::Identity<float_t>>;
using network = Network<Sequential>;
using lgl     = loss::Log_likelihood<float_t>;

void display_filtermaps(const tensor_t& output, const size_t in_height, const size_t in_width) {
  namedWindow("Display window", WINDOW_AUTOSIZE);
  Mat image(in_height, in_width, CV_8UC1);

  for (size_t batch = 0; batch < output.shape()[0]; ++batch) {
    for (size_t filter = 0; filter < output.shape()[1]; ++filter) {

      uchar *p = image.data;
      for (size_t i = 0; i < in_height; ++i) {
        for (size_t j = 0; j < in_width; ++j) {
          auto val = output.host_at(batch, filter, i, j);
          p[i * in_width + j] = val;
        }
      }

      imshow("Display window", image);
      waitKey(0);
    }
  }
}

static bool train_mnist() {
  size_t mnist_image_row  = 28;
  size_t mnist_image_col  = 28;
  size_t mnist_image_num = 60000;
  size_t in_width = 32;
  size_t in_height = 32;

  std::string path_to_data("/c3se/NOBACKUP/users/hacht/data/");

  std::vector<label_t> train_labels;
  tensor_t train_images({mnist_image_num, 1, in_height, in_width});
  float_t min = -1.0f;
  float_t max = 1.0f;
  train_images.fill(min);

  parse_mnist_images(path_to_data + "train-images.idx3-ubyte", &train_images, min, max, 2, 2);
  parse_mnist_labels(path_to_data + "train-labels.idx1-ubyte", &train_labels);

  // Display ----------------------------------------------------- //

  /*
  namedWindow("Display window", WINDOW_AUTOSIZE);
  Mat image(in_height, in_width, CV_8UC1);

  for (size_t samples = 0; samples < 100; ++samples) {
    uchar* p = image.data;
    for (int n = 0; n < in_height * in_width; ++n) {
      auto val = train_images.host_at_index(samples * in_height * in_width + n);
      p[n] = val;
    }

    imshow("Display window", image);
    waitKey(0);
  } */

  // -------------------------------------------------------------- //

  // Zero mean.
  train_images.add(-mean_value(train_images));
  size_t batch_size = 32;

  network net;
  net << conv(32, 32, 1, batch_size, 5, 6) << maxpool(28, 28, 6, batch_size) << conv(14, 14, 6, batch_size, 5, 16)
      << maxpool(10, 10, 16, batch_size) << conv(5, 5, 16, batch_size, 5, 120) << classy(120, 10, batch_size);

  Adam<float_t> a;

  net.test_mnist<lgl, Adam<float_t>>(a, train_images, train_labels, batch_size, 1);

  return true;
}


int main(int argc, char** argv) {
  if (train_mnist()) {
    return 0;
  }
  return -1;
}

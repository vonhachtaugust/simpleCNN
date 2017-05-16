//
// Created by hacht on 4/25/17.
//

#include <iostream>
#include "../../simpleCNN/simpleCNN.h"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui.hpp"

using namespace simpleCNN;
using namespace cv;

using dropout = Dropout_layer;
using conv    = Convolutional_layer;
using maxpool = Maxpooling_layer;
using fully   = Connected_layer;
using network = Network<Sequential>;
using adam    = Adam<float_t>;
using relu    = activation::ReLU;
using th      = activation::Tanh;
using softmax = loss::Softmax;

void display_filtermaps(const tensor_t& output, const size_t in_height, const size_t in_width) {
  namedWindow("Display window", WINDOW_AUTOSIZE);
  Mat image(in_height, in_width, CV_8UC1);

  for (size_t batch = 0; batch < output.shape()[0]; ++batch) {
    for (size_t filter = 0; filter < output.shape()[1]; ++filter) {
      uchar* p = image.data;
      for (size_t i = 0; i < in_height; ++i) {
        for (size_t j = 0; j < in_width; ++j) {
          auto val            = output.host_at(batch, filter, i, j);
          p[i * in_width + j] = val;
        }
      }

      imshow("Display window", image);
      waitKey(0);
    }
  }
}

static bool train_mnist(const size_t batch_size,
                        const size_t epoch,
                        const std::string weight_and_bias_file,
                        const std::string loss_file,
                        const std::string accuracy_file) {
  size_t mnist_image_row = 28;
  size_t mnist_image_col = 28;
  size_t mnist_image_num = 60000;
  size_t in_width        = 32;
  size_t in_height       = 32;
  size_t subset          = 1;

  std::string path_to_data("/c3se/NOBACKUP/users/hacht/data/");

  tensor_t train_labels({mnist_image_num / subset, 1, 1, 1});
  tensor_t train_images({mnist_image_num / subset, 1, in_height, in_width});
  float_t min = -1.0f;
  float_t max = 1.0f;
  train_images.fill(min);

  parse_mnist_images(path_to_data + "train-images.idx3-ubyte", &train_images, min, max, 2, 2, subset);
  parse_mnist_labels(path_to_data + "train-labels.idx1-ubyte", &train_labels, subset);

  // Display ----------------------------------------------------- //

  /*
  namedWindow("Display window", WINDOW_AUTOSIZE);
  Mat image(in_height, in_width, CV_8UC1);

  for (size_t sample = 0; sample < 100; ++sample) {
    uchar* p = image.data;
    for (size_t n = 0; n < in_height * in_width; ++n) {
      auto val = train_images.host_at_index(sample * in_height * in_width + n);
      p[n] = val;
    }

    print(train_labels.host_at_index(sample));
    imshow("Display window", image);
    waitKey(0);
  } */

  // -------------------------------------------------------------- //

  // Zero mean unit variance

  zero_mean_unit_variance(train_images, true, "mean_std16052017.txt");
  size_t minibatch_size = batch_size;
  size_t epochs         = epoch;

  // create callback
  auto on_enumerate_minibatch = [&](time_t t) {
    std::cout << (float_t)(clock() - t) / CLOCKS_PER_SEC << "s elapsed." << std::endl;
  };

  auto on_enumerate_epoch = [&](size_t epoch) { std::cout << epoch << std::endl; };

  network net;
  net << conv(32, 32, 1, minibatch_size, 5, 6) << relu() << maxpool(28, 28, 6, minibatch_size)
      << conv(14, 14, 6, minibatch_size, 5, 16) << relu() << maxpool(10, 10, 16, minibatch_size)
      << conv(5, 5, 16, minibatch_size, 5, 120) << relu() << dropout({minibatch_size, 120, 1, 1}, 0.5)
      << fully(120, 10, minibatch_size) << softmax();

  adam a;
  net.train<adam>(a, train_images, train_labels, minibatch_size, epochs, on_enumerate_minibatch, on_enumerate_epoch, weight_and_bias_file,
                  loss_file, accuracy_file, true);

  return true;
}

int main(int argc, char* argv[]) {
  if (argc < 6) {
    print("To few arguments, expected five.");
    return -1;
  }

  size_t batch_size = atoi(argv[1]);
  size_t epoch      = atoi(argv[2]);

  if (train_mnist(batch_size, epoch, argv[3], argv[4], argv[5])) {
    return 0;
  }
  return -1;
}

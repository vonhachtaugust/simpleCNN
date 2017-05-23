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
using relu    = Activation_layer;
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

static bool train_mnist(const size_t batch_size, const size_t epoch) {
  /** Mnist specific parameters  */
  size_t mnist_image_row = 28;
  size_t mnist_image_col = 28;
  size_t mnist_image_num = 60000;
  size_t in_width        = 28;
  size_t in_height       = 28;
  size_t subset          = 3;

  /** Path to the mnist data files */
  std::string path_to_data("/c3se/NOBACKUP/users/hacht/data/");

  /** Parse mnist */
  tensor_t labels({mnist_image_num / subset, 1, 1, 1});
  tensor_t images({mnist_image_num / subset, 1, in_height, in_width});

  float_t min = -1.0f;
  float_t max = 1.0f;
  images.fill(min);

  parse_mnist_images(path_to_data + "train-images.idx3-ubyte", &images, min, max, 0, 0, subset);
  parse_mnist_labels(path_to_data + "train-labels.idx1-ubyte", &labels, subset);

  /** Pre-processing */
  std::vector<float_t> mean_and_std = zero_mean_unit_variance(images);
  size_t minibatch_size             = batch_size;
  size_t epochs                     = epoch;
  mean_and_std.push_back(float_t(batch_size));
  mean_and_std.push_back(float_t(epoch));

  /** Split data into a training and validation set */
  float_t training_validation_split_ratio = 0.75;
  size_t training_set_size                = (mnist_image_num / subset) * training_validation_split_ratio;
  size_t validation_set_size = std::floor((mnist_image_num / subset) * (1 - training_validation_split_ratio) + 0.5);

  tensor_t train_images({training_set_size, 1, in_height, in_width});
  tensor_t train_labels({training_set_size, 1, 1, 1});
  tensor_t validation_images({validation_set_size, 1, in_height, in_width});
  tensor_t validation_labels({validation_set_size, 1, 1, 1});

  split_training_validation(images, train_images, validation_images, training_validation_split_ratio);
  split_training_validation(labels, train_labels, validation_labels, training_validation_split_ratio);

  /** Display ----------------------------------------------------- */

  // Remember to remove zero mean unit variance

  /*
  namedWindow("Display window", WINDOW_AUTOSIZE);
  Mat image(in_height, in_width, CV_8UC1);

  for (size_t sample = 0; sample < training_set_size; ++sample) {
    uchar* p = image.data;
    for (size_t n = 0; n < in_height * in_width; ++n) {
      auto val = train_images.host_at_index(sample * in_height * in_width + n);
      p[n] = val;
    }

    print(train_labels.host_at_index(sample), "train: " + std::to_string(sample));
    imshow("Display window", image);
    waitKey(0);
  }

  for (size_t sample = 0; sample < validation_set_size; ++sample) {
    uchar* p = image.data;
    for (size_t n = 0; n < in_height * in_width; ++n) {
      auto val = validation_images.host_at_index(sample * in_height * in_width + n);
      p[n] = val;
    }

    print(validation_labels.host_at_index(sample), "valid: " + std::to_string(sample));
    imshow("Display window", image);
    waitKey(0);
  }
   */

  /** -------------------------------------------------------------- */

  /** Call-back for clocking */
  auto on_enumerate_minibatch = [&](time_t t) {
    std::cout << (float_t)(clock() - t) / CLOCKS_PER_SEC << "s elapsed." << std::endl;
  };

  auto on_enumerate_epoch = [&](size_t epoch) { std::cout << epoch + 1 << std::endl; };

  /** Define network architecture and optimizer */
  network net;
  // net << conv(32, 32, 1, minibatch_size, 5, 6) << relu() << maxpool(28, 28, 6, minibatch_size)
  //    << conv(14, 14, 6, minibatch_size, 5, 16) << relu() << maxpool(10, 10, 16, minibatch_size)
  //    << conv(5, 5, 16, minibatch_size, 5, 120) << relu() << dropout({minibatch_size, 120, 1, 1}, 0.5)
  //    << fully(120, 10, minibatch_size) << softmax();
  float_t dropout_rate = 0.75;

  /* GPU - 21.56s */
  net << conv(28, 28, 1, minibatch_size, 5, 32, 1, 2, true, core::backend_t::gpu) << relu(core::activation_t::relu, core::backend_t::gpu)
      << maxpool(28, 28, 32, minibatch_size, 2, 2, 2, 2, core::backend_t::gpu)
      << conv(14, 14, 32, minibatch_size, 5, 64, 1, 2, true, core::backend_t::gpu) << relu(core::activation_t::relu, core::backend_t::gpu)
      << maxpool(14, 14, 64, minibatch_size, 2, 2, 2, 2, core::backend_t::gpu)
      << fully(7 * 7 * 64, 1024, minibatch_size, true, core::backend_t::gpu) << relu(core::activation_t::relu, core::backend_t::gpu)
      << dropout(dropout_rate, core::backend_t::gpu)
      << fully(1024, 10, minibatch_size, true, core::backend_t::gpu) << softmax();


  /* CPU - 902.74s
  net << conv(28, 28, 1, minibatch_size, 5, 32, 1, 2, true) << relu(core::activation_t::relu)
      << maxpool(28, 28, 32, minibatch_size, 2, 2, 2, 2)
      << conv(14, 14, 32, minibatch_size, 5, 64, 1, 2, true) << relu(core::activation_t::relu)
      << maxpool(14, 14, 64, minibatch_size, 2, 2, 2, 2)
      << fully(7 * 7 * 64, 1024, minibatch_size, true) << relu(core::activation_t::relu)
      << dropout(dropout_rate)
      << fully(1024, 10, minibatch_size, true) << softmax();
  */
  adam a;

  /** Train and save results */
  net.train<adam>(a, train_images, train_labels, validation_images, validation_labels, minibatch_size, epochs,
                  on_enumerate_minibatch, on_enumerate_epoch, true);
  net.save_results(mean_and_std);

  return true;
}

int main(int argc, char* argv[]) {
  size_t expect = 3;
  if (argc < expect) {
    print("To few arguments, expected " + std::to_string(expect));
    return -1;
  }

  /** program expects a batch size and an epoch size as command line argument */
  size_t batch_size = atoi(argv[1]);
  size_t epoch      = atoi(argv[2]);

  /** Let's go! */
  if (train_mnist(batch_size, epoch)) {
    return 0;
  }

  return -1;
}

//
// Created by hacht on 4/25/17.
//

#include <iostream>
#include "../../simpleCNN/simpleCNN.h"

using namespace simpleCNN;

using dropout = Dropout_layer;
using conv    = Convolutional_layer;
using maxpool = Maxpooling_layer;
using fully   = Connected_layer;
using network = Network<Sequential>;
using adam    = Adam<float_t>;
using relu    = Activation_layer;
using softmax = loss::Softmax;

static bool train_mnist(const size_t batch_size, const size_t epoch, const std::string data, const std::string result) {
  /** Mnist specific parameters  */
  size_t mnist_image_row = 28;
  size_t mnist_image_col = 28;
  size_t mnist_image_num = 60000;
  size_t mnist_test_num  = 10000;
  size_t in_width        = 28;
  size_t in_height       = 28;
  size_t subset          = 1;

  /** Path to the mnist data files */
  // std::string path_to_data("/c3se/NOBACKUP/users/hacht/data/");

  /** Parse mnist */
  tensor_t labels({mnist_image_num / subset, 1, 1, 1});
  tensor_t images({mnist_image_num / subset, 1, in_height, in_width});
  tensor_t test_images({mnist_test_num / subset, 1, in_height, in_width});
  tensor_t test_labels({mnist_test_num, 1, 1, 1});

  float_t min = -1.0f;
  float_t max = 1.0f;
  images.fill(min);

  parse_mnist_images(data + "/train-images.idx3-ubyte", &images, min, max, 0, 0, subset);
  parse_mnist_labels(data + "/train-labels.idx1-ubyte", &labels, subset);
  parse_mnist_images(data + "/t10k-images.idx3-ubyte", &test_images, min, max, 0, 0, subset);
  parse_mnist_labels(data + "/t10k-labels.idx1-ubyte", &test_labels);


  /** Pre-processing */
  std::vector<float_t> mean_and_std = zero_mean_unit_variance(images);
  size_t minibatch_size             = batch_size;
  size_t epochs                     = epoch;
  mean_and_std.push_back(float_t(batch_size));
  mean_and_std.push_back(float_t(epoch));

  zero_mean_unit_variance(test_images, mean_and_std[0], mean_and_std[1]);

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

  
  
  /*
  for (size_t sample = 0; sample < 100; sample++) {
    auto img = test_images.subView({sample}, {1, 1, in_height, in_width});
    auto lab = test_labels.subView({sample}, {1, 1, 1, 1});

    display_gray_image(img, *lab.host_begin(), mean_and_std.at(0), mean_and_std.at(1));
  } */

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

  /* GPU - 21.56s 429MiB */
  net << conv(28, 28, 1, minibatch_size, 5, 32, 1, 2, true, core::backend_t::gpu, true)
      << relu(core::activation_t::relu, core::backend_t::gpu)
      << maxpool(28, 28, 32, minibatch_size, 2, 2, 2, 2, core::backend_t::gpu)
      << conv(14, 14, 32, minibatch_size, 5, 64, 1, 2, true, core::backend_t::gpu, true)
      << relu(core::activation_t::relu, core::backend_t::gpu)
      << maxpool(14, 14, 64, minibatch_size, 2, 2, 2, 2, core::backend_t::gpu)
      << fully(7 * 7 * 64, 1024, minibatch_size, true, core::backend_t::gpu, true)
      << relu(core::activation_t::relu, core::backend_t::gpu)
      << dropout(dropout_rate)
      << fully(1024, 10, minibatch_size, true, core::backend_t::gpu, true)
      << softmax();

  /* CPU - 902.74s
  net << conv(28, 28, 1, minibatch_size, 5, 32, 1, 2, true) << relu(core::activation_t::relu)
   << maxpool(28, 28, 32, minibatch_size)
   << conv(14, 14, 32, minibatch_size, 5, 64, 1, 2, true) << relu(core::activation_t::relu)
   << maxpool(14, 14, 64, minibatch_size)
   << fully(7 * 7 * 64, 1024, minibatch_size, true) << relu(core::activation_t::relu)
   << dropout(dropout_rate)
   << fully(1024, 10, minibatch_size, true) << softmax();
   */

  adam a;
  /** Train and save results */
  //net.train<adam>(a, train_images, train_labels, validation_images, validation_labels, minibatch_size, epochs,
  //                on_enumerate_minibatch, on_enumerate_epoch, true);
  net.save_results(mean_and_std, result);
  //net.test_network(test_images, test_labels, minibatch_size, 10, result);

  return true;
}

int main(int argc, char* argv[]) {
  size_t expect = 5;
  if (argc < expect) {
    print("To few arguments, expected " + std::to_string(expect));
    return -1;
  }

  /** program expects a batch size and an epoch size as command line argument */
  size_t batch_size = atoi(argv[1]);
  size_t epoch      = atoi(argv[2]);
  std::string data  = argv[3];
  std::string result = argv[4];

  /** Let's go! */
  if (train_mnist(batch_size, epoch, data, result)) {
    return 0;
  }

  return -1;
}

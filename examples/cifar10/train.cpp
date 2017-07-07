//
// Created by hacht on 5/29/17.
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

static bool train_cifar(const size_t batch_size, const size_t epoch, const std::string data, const std::string result) {
  /** Cifar10 specific parameters  */
  size_t cifar_image_row  = 32;
  size_t cifar_image_col  = 32;
  size_t cifar_image_ch   = 3;
  size_t cifar_image_num  = 50000;
  size_t cifar_batch_size = 10000;
  size_t cifar_test_num   = 10000;
  size_t in_width         = 32;
  size_t in_height        = 32;
  size_t in_ch            = 3;
  size_t subset           = 1;

  /** Parse cifar */
  tensor_t labels({cifar_image_num / subset, 1, 1, 1});
  tensor_t images({cifar_image_num / subset, in_ch, in_height, in_width});
  tensor_t test_images({cifar_test_num, in_ch, in_height, in_width});
  tensor_t test_labels({cifar_test_num, 1, 1, 1});

  float_t min = -1.0f;  // 0
  float_t max = 1.0f;   // 255
  images.fill(min);

  for (size_t i = 1; i <= 5; ++i) {
    parse_cifar10(data + "/data_batch_" + std::to_string(i) + ".bin", &images, &labels, min, max, 0, 0,
                  (i - 1) * cifar_batch_size * cifar_image_ch * cifar_image_row * cifar_image_col,
                  (i - 1) * cifar_batch_size, subset);
  }

  parse_cifar10(data + "/test_batch.bin", &test_images, &test_labels, min, max, 0, 0);

  /** Pre-processing */
  std::vector<float_t> mean = zero_mean(images);
  size_t minibatch_size     = batch_size;
  size_t epochs             = epoch;
  mean.push_back(float_t(batch_size));
  mean.push_back(float_t(epoch));

  zero_mean(test_images, mean[0]);

  /** Split data into a training and validation set */
  float_t training_validation_split_ratio = 0.9;
  size_t training_set_size                = (cifar_image_num / subset) * training_validation_split_ratio;
  size_t validation_set_size = std::floor((cifar_image_num / subset) * (1 - training_validation_split_ratio) + 0.5);

  tensor_t train_images({training_set_size, in_ch, in_height, in_width});
  tensor_t train_labels({training_set_size, 1, 1, 1});
  tensor_t validation_images({validation_set_size, in_ch, in_height, in_width});
  tensor_t validation_labels({validation_set_size, 1, 1, 1});

  split_training_validation(images, train_images, validation_images, training_validation_split_ratio);
  split_training_validation(labels, train_labels, validation_labels, training_validation_split_ratio);

  /** -------------------------------------------------------------- */

  /** Call-back for clocking */
  auto on_enumerate_minibatch = [&](time_t t) {
    std::cout << (float_t)(clock() - t) / CLOCKS_PER_SEC << "s elapsed." << std::endl;
  };

  auto on_enumerate_epoch = [&](size_t epoch) { std::cout << epoch + 1 << std::endl; };

  /** Define network architecture and optimizer */
  network net;
  float_t dropout_rate = 0.5;

  /* GPU - 1 */
  net << conv(32, 32, 3, minibatch_size, 5, 3 * 32, 1, 2, true, core::backend_t::gpu)
      << relu(core::activation_t::relu, core::backend_t::gpu)
      << maxpool(32, 32, 3 * 32, minibatch_size, 2, 2, 2, 2, core::backend_t::gpu)
      << conv(16, 16, 3 * 32, minibatch_size, 5, 3 * 64, 1, 2, true, core::backend_t::gpu)
      << relu(core::activation_t::relu, core::backend_t::gpu)
      << maxpool(16, 16, 3 * 64, minibatch_size, 2, 2, 2, 2, core::backend_t::gpu)
      << fully(8 * 8 * 3 * 64, 2 * 1024, minibatch_size, true, core::backend_t::gpu)
      << relu(core::activation_t::relu, core::backend_t::gpu) << dropout(dropout_rate)
      << fully(2 * 1024, 2 * 1024, minibatch_size, true, core::backend_t::gpu)
      << relu(core::activation_t::relu, core::backend_t::gpu) << dropout(dropout_rate)
      << fully(2 * 1024, 10, minibatch_size, true, core::backend_t::gpu) << softmax();

  /* GPU - 2
  net << conv(32, 32, 3, minibatch_size, 5, 160, 1, 2, true, core::backend_t::gpu, true)
      << relu(core::activation_t::relu, core::backend_t::gpu)
      << conv(32, 32, 160, minibatch_size, 5, 160, 1, 2, true, core::backend_t::gpu, false)
      << relu(core::activation_t::relu, core::backend_t::gpu)
      << maxpool(32, 32, 160, minibatch_size, 2, 2, 2, 2, core::backend_t::gpu)

      << conv(16, 16, 160, minibatch_size, 5, 160, 1, 2, true, core::backend_t::gpu, false)
      << relu(core::activation_t::relu, core::backend_t::gpu)
      << conv(16, 16, 160, minibatch_size, 5, 160, 1, 2, true, core::backend_t::gpu, false)
      << relu(core::activation_t::relu, core::backend_t::gpu)
      << maxpool(16, 16, 160, minibatch_size, 2, 2, 2, 2, core::backend_t::gpu)

      << fully(8 * 8 * 160, 2048, minibatch_size, true, core::backend_t::gpu, false)
      << relu(core::activation_t::relu, core::backend_t::gpu)
      << dropout(dropout_rate)
      << fully(2048, 2048, minibatch_size, true, core::backend_t::gpu, false)
      << relu(core::activation_t::relu, core::backend_t::gpu)
      << dropout(dropout_rate)
      << fully(2048, 10, minibatch_size, true, core::backend_t::gpu, false)
      << softmax();
  */

  /* GPU -  3
  net << conv(32, 32, 3, minibatch_size, 3, 96, 1, 1, true, core::backend_t::gpu, true)
      << relu(core::activation_t::relu, core::backend_t::gpu)
      << conv(32, 32, 96, minibatch_size, 3, 96, 1, 1, true, core::backend_t::gpu, false)
      << relu(core::activation_t::relu, core::backend_t::gpu)
      << maxpool(32, 32, 96, minibatch_size, 2, 2, 2, 2, core::backend_t::gpu)

      << conv(16, 16, 96, minibatch_size, 3, 192, 1, 1, true, core::backend_t::gpu, false)
      << relu(core::activation_t::relu, core::backend_t::gpu)
      << conv(16, 16, 192, minibatch_size, 3, 192, 1, 1, true, core::backend_t::gpu, false)
      << relu(core::activation_t::relu, core::backend_t::gpu)
      << maxpool(16, 16, 192, minibatch_size, 2, 2, 2, 2, core::backend_t::gpu)

      << fully(8 * 8 * 192, 2048, minibatch_size, true, core::backend_t::gpu, false)
      << relu(core::activation_t::relu, core::backend_t::gpu)
      << dropout(dropout_rate)
      << fully(2048, 2048, minibatch_size, true, core::backend_t::gpu, false)
      << relu(core::activation_t::relu, core::backend_t::gpu)
      << dropout(dropout_rate)
      << fully(2048, 10, minibatch_size, true, core::backend_t::gpu, false)
      << softmax();
  */

  /* GPU - 4
  net << conv(32, 32, 3, minibatch_size, 5, 32, 1, 2, true, core::backend_t::gpu, true)
      << relu(core::activation_t::relu, core::backend_t::gpu)
      << maxpool(32, 32, 32, minibatch_size, 2, 2, 1, 1, core::backend_t::gpu)

      << conv(31, 31, 32, minibatch_size, 5, 32, 1, 2, true, core::backend_t::gpu, false)
      << relu(core::activation_t::relu, core::backend_t::gpu)
      << maxpool(31, 31, 32, minibatch_size, 2, 2, 1, 1, core::backend_t::gpu)

      << conv(30, 30, 32, minibatch_size, 5, 32, 1, 2, true, core::backend_t::gpu, false)
      << relu(core::activation_t::relu, core::backend_t::gpu)
      << maxpool(30, 30, 32, minibatch_size, 2, 2, 1, 1, core::backend_t::gpu)

      << conv(29, 29, 32, minibatch_size, 5, 64, 1, 2, true, core::backend_t::gpu, false)
      << relu(core::activation_t::relu, core::backend_t::gpu)
      << maxpool(29, 29, 64, minibatch_size, 2, 2, 1, 1, core::backend_t::gpu)

      << conv(28, 28, 64, minibatch_size, 5, 64, 1, 2, true, core::backend_t::gpu, false)
      << relu(core::activation_t::relu, core::backend_t::gpu)
      << maxpool(28, 28, 64, minibatch_size, 2, 2, 1, 1, core::backend_t::gpu)

      << conv(27, 27, 64, minibatch_size, 5, 64, 1, 2, true, core::backend_t::gpu, false)
      << relu(core::activation_t::relu, core::backend_t::gpu)
      << maxpool(27, 27, 64, minibatch_size, 2, 2, 1, 1, core::backend_t::gpu)

      << conv(26, 26, 96, minibatch_size, 5, 96, 1, 2, true, core::backend_t::gpu, false)
      << relu(core::activation_t::relu, core::backend_t::gpu)
      << maxpool(26, 26, 96, minibatch_size, 2, 2, 1, 1, core::backend_t::gpu)

      << conv(25, 25, 96, minibatch_size, 5, 96, 1, 2, true, core::backend_t::gpu, false)
      << relu(core::activation_t::relu, core::backend_t::gpu)
      << maxpool(25, 25, 96, minibatch_size, 2, 2, 1, 1, core::backend_t::gpu)

      << conv(24, 24, 96, minibatch_size, 5, 96, 1, 2, true, core::backend_t::gpu, false)
      << relu(core::activation_t::relu, core::backend_t::gpu)
      << maxpool(24, 24, 96, minibatch_size, 2, 2, 2, 2, core::backend_t::gpu)

      << fully(12 * 12 * 96, 2046, minibatch_size, true, core::backend_t::gpu, false)
      << relu(core::activation_t::relu, core::backend_t::gpu)
      << dropout(dropout_rate)
      << fully(2046, 2046, minibatch_size, true, core::backend_t::gpu, false)
      << relu(core::activation_t::relu, core::backend_t::gpu)
      << dropout(dropout_rate)
      << fully(2046, 10, minibatch_size, true, core::backend_t::gpu, false)
      << softmax();
  */

  /* GPU - 5
  net << conv(32, 32, 3, minibatch_size, 3, 64, 1, 1, true, core::backend_t::gpu, true)
      << relu(core::activation_t::relu, core::backend_t::gpu)
      << conv(32, 32, 64, minibatch_size, 3, 64, 1, 1, true, core::backend_t::gpu, false)
      << relu(core::activation_t::relu, core::backend_t::gpu)
      << maxpool(32, 32, 64, minibatch_size, 2, 2, 1, 1, core::backend_t::gpu)

      << conv(31, 31, 64, minibatch_size, 3, 128, 1, 1, true, core::backend_t::gpu, true)
      << relu(core::activation_t::relu, core::backend_t::gpu)
      << conv(31, 31, 128, minibatch_size, 3, 128, 1, 1, true, core::backend_t::gpu, false)
      << relu(core::activation_t::relu, core::backend_t::gpu)
      << maxpool(31, 31, 128, minibatch_size, 2, 2, 1, 1, core::backend_t::gpu)

      << conv(30, 30, 128, minibatch_size, 3, 128, 1, 1, true, core::backend_t::gpu, false)
      << relu(core::activation_t::relu, core::backend_t::gpu)
      << maxpool(30, 30, 128, minibatch_size, 2, 2, 2, 2, core::backend_t::gpu)

      << fully(15 * 15 * 128, 2048, minibatch_size, true, core::backend_t::gpu, false)
      << relu(core::activation_t::relu, core::backend_t::gpu)
      << dropout(dropout_rate)
      << fully(2048, 2048, minibatch_size, true, core::backend_t::gpu, false)
      << relu(core::activation_t::relu, core::backend_t::gpu)
      << dropout(dropout_rate)
      << fully(2048, 2048, minibatch_size, true, core::backend_t::gpu, false)
      << relu(core::activation_t::relu, core::backend_t::gpu)
      << dropout(dropout_rate)
      << fully(2048, 10, minibatch_size, true, core::backend_t::gpu, false)
      << softmax();
  */

  /* GPU - 6
 net << conv(32, 32, 3, minibatch_size, 3, 128, 1, 1, true, core::backend_t::gpu, true)
     << relu(core::activation_t::relu, core::backend_t::gpu)
     << conv(32, 32, 128, minibatch_size, 3, 128, 1, 1, true, core::backend_t::gpu, false)
     << relu(core::activation_t::relu, core::backend_t::gpu)
     << maxpool(32, 32, 128, minibatch_size, 2, 2, 1, 1, core::backend_t::gpu)

     << conv(31, 31, 128, minibatch_size, 3, 128, 1, 1, true, core::backend_t::gpu, true)
     << relu(core::activation_t::relu, core::backend_t::gpu)
     << conv(31, 31, 128, minibatch_size, 3, 128, 1, 1, true, core::backend_t::gpu, false)
     << relu(core::activation_t::relu, core::backend_t::gpu)
     << maxpool(31, 31, 128, minibatch_size, 2, 2, 1, 1, core::backend_t::gpu)

     << conv(30, 30, 128, minibatch_size, 3, 128, 1, 1, true, core::backend_t::gpu, false)
     << relu(core::activation_t::relu, core::backend_t::gpu)
     << maxpool(30, 30, 128, minibatch_size, 2, 2, 2, 2, core::backend_t::gpu)

     << fully(15 * 15 * 128, 4096, minibatch_size, true, core::backend_t::gpu, false)
     << relu(core::activation_t::relu, core::backend_t::gpu)
     << dropout(dropout_rate)
     << fully(4096, 4096, minibatch_size, true, core::backend_t::gpu, false)
     << relu(core::activation_t::relu, core::backend_t::gpu)
     << dropout(dropout_rate)
     << fully(4096, 4096, minibatch_size, true, core::backend_t::gpu, false)
     << relu(core::activation_t::relu, core::backend_t::gpu)
     << dropout(dropout_rate)
     << fully(4096, 10, minibatch_size, true, core::backend_t::gpu, false)
     << softmax();
     */

  adam a;
  /** Train and save results */
  net.train<adam>(a, train_images, train_labels, validation_images, validation_labels, minibatch_size, epochs,
                  on_enumerate_minibatch, on_enumerate_epoch, true);
  net.save_results(mean, result);
  net.test_network(test_images, test_labels, minibatch_size, 10, result);

  return true;
}

int main(int argc, char* argv[]) {
  size_t expect = 5;
  if (argc < expect) {
    print("To few arguments, expted" + std::to_string(expect) + "\n");
    print("Usage: ./example_cifar10 batch_size epoch data_path store_results\n");
    print("Example usage: ./example_cifar10 100 50 ~/data ~/results\n");
    return -1;
  }

  /** program expects a batch size and an epoch size as command line argument */
  size_t batch_size  = atoi(argv[1]);  // use 50 or 100 etc
  size_t epoch       = atoi(argv[2]);
  std::string data   = argv[3];
  std::string result = argv[4];

  /** Let's go! */
  if (train_cifar(batch_size, epoch, data, result)) {
    return 0;
  }

  return -1;
}

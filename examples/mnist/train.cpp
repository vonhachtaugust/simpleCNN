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
using maxpool = Maxpooling_layer<>;
using fully   = Connected_layer<>;
using classy  = Connected_layer<float_t, activation::Softmax<float_t>>;
using network = Network<Sequential>;
using lgl     = loss::Log_likelihood<float_t>;
using adam    = Adam<float_t>;

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

  tensor_t train_labels({mnist_image_num, 1, 1, 1});
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

  // Zero mean.
  train_images.add(-mean_value(train_images));
  size_t minibatch_size = 10;
  size_t epochs = 1;


  // create callback
  auto on_enumerate_minibatch = [&](time_t t){
    std::cout << (float_t) (clock() - t) / CLOCKS_PER_SEC << "s elapsed." << std::endl;
  };

  auto on_enumerate_epoch = [&](size_t epoch){
    std::cout << epoch << std::endl;
  };

  network net;
  net << conv(32, 32, 1, minibatch_size, 5, 6) << maxpool(28, 28, 6, minibatch_size) << conv(14, 14, 6, minibatch_size, 5, 16)
      << maxpool(10, 10, 16, minibatch_size) << conv(5, 5, 16, minibatch_size, 5, 120) << dropout({minibatch_size, 120, 1, 1}, 0.5) << classy(120, 10, minibatch_size);

  adam a;
  net.train<lgl, adam>(a, train_images, train_labels, minibatch_size, epochs, on_enumerate_minibatch, on_enumerate_epoch, true);
  return true;
}


int main(int argc, char** argv) {
  if (train_mnist()) {
    return 0;
  }
  return -1;
}

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

static void train_mnist() {
  size_t mnist_image_row  = 28;
  size_t mnist_image_col  = 28;
  size_t mnist_image_size = 60000;

  std::string path_to_data("/c3se/NOBACKUP/users/hacht/data/");

  std::vector<label_t> train_labels;
  tensor_t train_images({mnist_image_size, 1, mnist_image_row, mnist_image_col});

  parse_mnist_images(path_to_data + "train-images.idx3-ubyte", &train_images, -1.0, 1.0, 0, 0);
  parse_mnist_labels(path_to_data + "train-labels.idx1-ubyte", &train_labels);

  size_t batch_size = 1;
  tensor_t test_batch({batch_size, 1, mnist_image_row, mnist_image_col});

  size_t size = mnist_image_col * mnist_image_row;
  for (size_t n = 0; n < batch_size * mnist_image_col * mnist_image_row; ++n) {
    test_batch.host_at_index(n) = train_images.host_at_index(n);
  }

  /*
  namedWindow("Display window", WINDOW_AUTOSIZE);
  Mat image(mnist_image_row, mnist_image_col, CV_8UC1);

  for (size_t samples = 0; samples < batch_size; ++samples) {
    uchar* p = image.data;
    for (int n = 0; n < mnist_image_row * mnist_image_col; ++n) {
      p[n] = test_batch.host_at_index(samples * mnist_image_row * mnist_image_col + n);
    }

    imshow("Display window", image);
    waitKey(0);
  } */

  network net;

  // Conv(in_w, in_h, in_c, b_size, filter_size, out_c, stride, padding, has_bias)
  // Max(in_w, in_h, in_c, b_size, pool_size, stride_size)
  // fc(in_dim, out_dim, batch_size, has_bias)

  net << conv(28, 28, 1, 1, 5, 6, 1, 2, true) << maxpool(28, 28, 6, 1) << conv(14, 14, 6, 1, 5, 16)
      << maxpool(10, 10, 16, 1) << conv(5, 5, 16, 1, 5, 120) << classy(120, 10, 1);

  tensor_t output = net.test(test_batch);
  print(output, "Output");
}

int main(int argc, char** argv) { train_mnist(); }
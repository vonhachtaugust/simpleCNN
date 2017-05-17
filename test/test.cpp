//
// Created by hacht on 3/16/17.
//

#include "../simpleCNN/simpleCNN.h"
#include "gtest/gtest.h"

#include "test_activation_layer.h"
#include "test_connected_layer.h"
#include "test_convolutional_layer.h"
#include "test_cuda.h"
#include "test_dropout_layer.h"
#include "test_loss_layer.h"
#include "test_math_functions.h"
#include "test_maxpooling_layer.h"
#include "test_network.h"
#include "test_optimizer.h"
#include "test_serialization.h"
#include "test_tensor.h"
#include "test_tensor_multiplication.h"
#include "test_weight_init.h"

using namespace std;

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

//
// Created by hacht on 3/16/17.
//

#include "../simpleCNN/simpleCNN.h"
#include "gtest/gtest.h"

#include "test_convolutional_layer.h"
#include "test_maxpooling_layer.h"

using namespace std;

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

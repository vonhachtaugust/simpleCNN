//
// Created by hacht on 3/16/17.
//

#include "../simpleCNN/simpleCNN.h"
#include "gtest/gtest.h"

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  std::cout << "Running all tests!" << std::endl;
  return RUN_ALL_TESTS();
}

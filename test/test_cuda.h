//
// Created by hacht on 5/1/17.
//

#pragma once

#include "../simpleCNN/simpleCNN.h"
#include "gtest/gtest.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>
#include "../third_party/cudnn/include/cudnn.h"

#include "time.h"

namespace simpleCNN {

#define checkCudaErrors(status)             \
  do {                                      \
    std::stringstream _error;               \
    if (status != 0) {                      \
      _error << "Cuda failure: " << status; \
      FatalError(_error.str());             \
    }                                       \
  } while (0)

#define FatalError(s)                                                 \
  do {                                                                \
    std::stringstream _where, _message;                               \
    _where << __FILE__ << ':' << __LINE__;                            \
    _message << std::string(s) + "\n" << __FILE__ << ':' << __LINE__; \
    std::cerr << _message.str() << "\nAborting...\n";                 \
    cudaDeviceReset();                                                \
    exit(1);                                                          \
  } while (0)

  float_t* cuda_make_array(float_t* x, size_t n) {
    float_t* x_gpu;
    checkCudaErrors(cudaMalloc((void**)&x_gpu, sizeof(float_t) * n));
    if (x) {
      checkCudaErrors(cudaMemcpy(x_gpu, x, sizeof(float_t) * n, cudaMemcpyHostToDevice));
    }

    if (!x_gpu) {
      simple_error("Cuda malloc failed\n");
    }

    return x_gpu;
  }

  void cuda_free(float_t* x_gpu) { checkCudaErrors(cudaFree(x_gpu)); }

  void cuda_push_array(float_t* x_gpu, float_t* x, size_t n) {
    checkCudaErrors(cudaMemcpy(x_gpu, x, sizeof(float_t) * n, cudaMemcpyHostToDevice));
  }

  void cuda_pull_array(float_t* x_gpu, float_t* x, size_t n) {
    checkCudaErrors(cudaMemcpy(x, x_gpu, sizeof(float_t) * n, cudaMemcpyDeviceToHost));
  }

  TEST(Cuda, print_device_info) {
    int nDevices;

    cudaGetDeviceCount(&nDevices);
    for (int i = 0; i < nDevices; i++) {
      cudaDeviceProp prop;
      cudaGetDeviceProperties(&prop, i);
      printf("Device Number: %d\n", i);
      printf("  Device name: %s\n", prop.name);
      printf("  Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
      printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
      printf("  Peak Memory Bandwidth (GB/s): %f\n\n", 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
    }
  }

#define DEFINE_int32(flag, default_value, description) const int FLAGS_##flag = (default_value)
#define DEFINE_uint64(flag, default_value, description) const unsigned long long FLAGS_##flag = (default_value)
#define DEFINE_bool(flag, default_value, description) const bool FLAGS_##flag = (default_value)
#define DEFINE_double(flag, default_value, description) const double FLAGS_##flag = (default_value)
#define DEFINE_string(flag, default_value, description) const std::string FLAGS_##flag ((default_value))

DEFINE_int32(gpu, 0, "The GPU ID to use");

  TEST(Cuda, gemm_gpu) {
    std::vector<float_t> x = {1, 2, 3};

    float_t* x_gpu = cuda_make_array(&x[0], x.size());

    float_t x_cpu[3];

    cuda_pull_array(x_gpu, x_cpu, x.size());

    for (size_t i = 0; i < 3; ++i) {
      ASSERT_EQ(x_cpu[i], x[i]);
    }

    cuda_free(x_gpu);
  }

  TEST(Cuda, cudnnConvolution) {
  cudnnHandle_t cudnnHandle;





}
}  // namespace simpleCNN
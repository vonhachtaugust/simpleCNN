//
// Created by hacht on 5/21/17.
//

#pragma once

#ifdef USE_CUDNN
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_device_runtime_api.h>
#include <cuda_fp16.h>
#include <cuda.h>


#include "../../third_party/cudnn/include/cudnn.h"
#include <cublas_v2.h>

#include "simple_error.h"



namespace simpleCNN {

// Error handling
// Adapted from the CUDNN classification code
// sample: https://developer.nvidia.com/cuDNN

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

#define checkCUDNN(status)                                        \
  do {                                                            \
    std::stringstream _error;                                     \
    if (status != CUDNN_STATUS_SUCCESS) {                         \
      _error << "CUDNN failure: " << cudnnGetErrorString(status); \
      FatalError(_error.str());                                   \
    }                                                             \
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

  int cuda_get_device() {
    int gpu_index = 0;
    checkCudaErrors(cudaGetDevice(&gpu_index));
    return gpu_index;
  }

  cudnnHandle_t cudnn_handle() {
    static int init[16] = {0};
    static cudnnHandle_t handle[16];
    int i = cuda_get_device();
    if (!init[i]) {
      cudnnCreate(&handle[i]);
      init[i] = 1;
    }
    return handle[i];
  }



  cublasHandle_t blas_handle()
  {
    static int init[16] = {0};
    static cublasHandle_t handle[16];
    int i = cuda_get_device();
    if(!init[i]) {
      cublasCreate(&handle[i]);
      init[i] = 1;
    }
    return handle[i];
  }
}

#endif
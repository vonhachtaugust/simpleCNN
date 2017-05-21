//
// Created by hacht on 4/5/17.
//

#pragma once

#include "params.h"

#ifdef USE_CUDNN
#include "../../../third_party/cudnn/include/cudnn.h"
#include "../../util/cuda_utils.h"
#endif

namespace simpleCNN {
  namespace core {

    class Con_params : public Params {
     public:
      // Input parameters
      size_t in_dim;
      size_t out_dim;
      size_t batch_size;
      bool has_bias;

      const Con_params& connected_params() const { return *this; }

#ifdef USE_CUDNN

      float_t* onevec;
      cudnnHandle_t cudnnHandle;
      cublasHandle_t cublasHandle;
      cudnnTensorDescriptor_t srcTensorDesc, dstTensorDesc;
      cudnnTensorDescriptor_t dsrcTensorDesc, ddstTensorDesc;

      void initialize_gpu_descriptors() {
        cudnnHandle  = cudnn_handle();
        cublasHandle = blas_handle();

        checkCUDNN(cudnnCreateTensorDescriptor(&srcTensorDesc));
        checkCUDNN(
          cudnnSetTensor4dDescriptor(srcTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, 1, in_dim, 1));

        checkCUDNN(cudnnCreateTensorDescriptor(&dstTensorDesc));
        checkCUDNN(
          cudnnSetTensor4dDescriptor(dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, 1, out_dim, 1));

        checkCudaErrors(cudaMalloc(&onevec, sizeof(float_t) * batch_size));
        initialized_gpu = true;
      }

      ~Con_params() {
        if (initialized_gpu) {
          //checkCudaErrors(cublasDestroy(cublasHandle));
          //checkCUDNN(cudnnDestroy(cudnnHandle));
          checkCUDNN(cudnnDestroyTensorDescriptor(srcTensorDesc));
          checkCUDNN(cudnnDestroyTensorDescriptor(dstTensorDesc));
          checkCudaErrors(cudaFree(onevec));
        }
      }
#endif

     private:
#ifdef USE_CUDNN
    bool initialized_gpu = false;
#endif
    };
  }
}
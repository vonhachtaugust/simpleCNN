//
// Created by hacht on 4/3/17.
//

#pragma once

#include "../../util/util.h"
#include "params.h"

#ifdef USE_CUDNN
#include "../../../third_party/cudnn/include/cudnn.h"
#include "../../util/cuda_utils.h"
#endif

namespace simpleCNN {
  namespace core {

    class Maxpooling_params : public Params {
     public:
      // Input parameters
      size_t input_width;
      size_t input_height;
      size_t in_channels;
      size_t batch_size;

      // filter parameters
      size_t pooling_size_x;
      size_t pooling_size_y;
      size_t stride_x;
      size_t stride_y;

      // Output parameters
      size_t output_width;
      size_t output_height;
      size_t out_channels;

      // Keep track of winning tile in forward pass
      tensor_t max_index;

      const Maxpooling_params& maxpool() const { return *this; }

      size_t pooling_size() const {
        if (pooling_size_x == pooling_size_y) {
          return pooling_size_x;
        }
        std::cout << pooling_size_x << "\t" << pooling_size_y << std::endl;
        throw simple_error(
          "Error: Filter sizes are different, therefore filter size is "
          "undefined");
      }

      size_t stride() const {
        if (stride_x == stride_y) {
          return stride_x;
        }

        std::cout << stride_x << "\t" << stride_y << std::endl;
        throw simple_error("Error: Stride sizes are different, therefore stride is undefined");
      }

#ifdef USE_CUDNN
      cudnnHandle_t cudnnHandle;
      cudnnTensorDescriptor_t srcTensorDesc, dstTensorDesc, dsrcTensorDesc, ddstTensorDesc;
      cudnnPoolingDescriptor_t poolDesc;

      void initalize_gpu_descriptors() {
        cudnnHandle  = cudnn_handle();

        checkCUDNN(cudnnCreateTensorDescriptor(&srcTensorDesc));
        checkCUDNN(cudnnSetTensor4dDescriptor(srcTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size,
                                              in_channels, input_height, input_width));
        checkCUDNN(cudnnCreateTensorDescriptor(&dsrcTensorDesc));
        checkCUDNN(cudnnSetTensor4dDescriptor(dsrcTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size,
                                              in_channels, input_height, input_width));

        checkCUDNN(cudnnCreateTensorDescriptor(&dstTensorDesc));
        checkCUDNN(cudnnSetTensor4dDescriptor(dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size,
                                              out_channels, output_height, output_width));

        checkCUDNN(cudnnCreateTensorDescriptor(&ddstTensorDesc));
        checkCUDNN(cudnnSetTensor4dDescriptor(ddstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size,
                                              out_channels, output_height, output_width));

        checkCUDNN(cudnnCreatePoolingDescriptor(&poolDesc));
        checkCUDNN(cudnnSetPooling2dDescriptor(poolDesc, CUDNN_POOLING_MAX, CUDNN_PROPAGATE_NAN, pooling_size_y,
                                               pooling_size_x, 0, 0, stride_y, stride_x));
        initialized_gpu = true;
      }

      ~Maxpooling_params() {
        if (initialized_gpu) {
          // checkCUDNN(cudnnDestroy(cudnnHandle));
          checkCUDNN(cudnnDestroyTensorDescriptor(srcTensorDesc));
          checkCUDNN(cudnnDestroyTensorDescriptor(dsrcTensorDesc));
          checkCUDNN(cudnnDestroyTensorDescriptor(dstTensorDesc));
          checkCUDNN(cudnnDestroyTensorDescriptor(ddstTensorDesc));
          checkCUDNN(cudnnDestroyPoolingDescriptor(poolDesc));
        }
      }

#endif
     private:
#ifdef  USE_CUDNN
      bool initialized_gpu = false;
#endif
    };
  }  // namespace core
}  // namespace simpleCNN

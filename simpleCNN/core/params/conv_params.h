//  // Created by hacht on 3/8/17.  //
#pragma once

#include "../../util/util.h"
#include "params.h"

#ifdef USE_CUDNN
#include "../../../third_party/cudnn/include/cudnn.h"
#include "../../util/cuda_utils.h"
#endif

namespace simpleCNN {
  namespace core { /**
         * @breif Convolution settings
         *
         * currently only supports 2D
                      convolution by s(tr/l)iding
         * window for each
                      depth/channel.
         */
    class Conv_params : public Params {
     public:
      // Input parameters
      size_t input_width;
      size_t input_height;
      size_t in_channels;
      size_t batch_size;

      // Filter parameters (num filters = out_channels)
      size_t filter_width;
      size_t filter_height;
      size_t horizontal_stride;
      size_t vertical_stride;
      size_t padding;
      bool has_bias;

      // Output parameters
      size_t output_width;
      size_t output_height;
      size_t out_channels;

      const Conv_params& conv() const {
        return *this;
      } /**
               * @breif common case of symmetric striding
               *
           */
      size_t stride() const {
        if (vertical_stride == horizontal_stride) {
          return vertical_stride;
        }
        std::cout << vertical_stride << "\t" << horizontal_stride << std::endl;
        throw simple_error("Error: Stride sizes are different, stride is undefined");
      } /**
               * @brief common case that filter size is symmetric
           *
               */
      size_t filter_size() const {
        if (filter_width == filter_height) {
          return filter_width;
        }
        std::cout << filter_width << "\t" << filter_height << std::endl;
        throw simple_error("Error: Filter sizes are different, therefore filter size is undefined");
      }

#ifdef USE_CUDNN
      cudnnHandle_t cudnnHandle;
      cublasHandle_t cublasHandle;
      cudnnTensorDescriptor_t srcTensorDesc, dstTensorDesc, biasDesc;
      cudnnTensorDescriptor_t dsrcTensorDesc, ddstTensorDesc, dbiasDesc;
      cudnnFilterDescriptor_t weightDesc, dweightDesc;
      cudnnConvolutionDescriptor_t convDesc;

      cudnnConvolutionFwdAlgo_t fw_algo;
      cudnnConvolutionBwdFilterAlgo_t bf_algo;
      cudnnConvolutionBwdDataAlgo_t bd_algo;

      size_t workspace_size;
      void* workspace = nullptr;

      void initalize_gpu_descriptors() {
        // Here we go ...
        cudnnHandle = cudnn_handle();
        cublasHandle = blas_handle();

        /** Input */
        checkCUDNN(cudnnCreateTensorDescriptor(&srcTensorDesc));
        checkCUDNN(cudnnSetTensor4dDescriptor(srcTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size,
                                              in_channels, input_height, input_width));

        /** Backpropr input gradient */
        checkCUDNN(cudnnCreateTensorDescriptor(&dsrcTensorDesc));
        checkCUDNN(cudnnSetTensor4dDescriptor(dsrcTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size,
                                              in_channels, input_height, input_width));

        /** Output */
        checkCUDNN(cudnnCreateTensorDescriptor(&dstTensorDesc));
        checkCUDNN(cudnnSetTensor4dDescriptor(dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size,
                                              out_channels, output_height, output_width));

        /** Backprop output gradient */
        checkCUDNN(cudnnCreateTensorDescriptor(&ddstTensorDesc));
        checkCUDNN(cudnnSetTensor4dDescriptor(ddstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size,
                                              out_channels, output_height, output_width));

        /** Weights */
        checkCUDNN(cudnnCreateFilterDescriptor(&weightDesc));
        checkCUDNN(cudnnSetFilter4dDescriptor(weightDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, out_channels,
                                              in_channels, filter_height, filter_width));

        /** dW */
        checkCUDNN(cudnnCreateFilterDescriptor(&dweightDesc));
        checkCUDNN(cudnnSetFilter4dDescriptor(dweightDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, out_channels,
                                              in_channels, filter_height, filter_width));

        /** Convolution specification */
        checkCUDNN(cudnnCreateConvolutionDescriptor(&convDesc));
        checkCUDNN(cudnnSetConvolution2dDescriptor(convDesc, padding, padding, vertical_stride, horizontal_stride, 1, 1,
                                                   CUDNN_CROSS_CORRELATION));

        /** Forward prop specification */
        checkCUDNN(cudnnGetConvolutionForwardAlgorithm(cudnnHandle, srcTensorDesc, weightDesc, convDesc, dstTensorDesc,
                                                       CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &fw_algo));

        /** Backward data specification */
        checkCUDNN(cudnnGetConvolutionBackwardDataAlgorithm(cudnnHandle, weightDesc, ddstTensorDesc, convDesc,
                                                            dsrcTensorDesc, CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
                                                            0, &bd_algo));

        /** Backward filter specification */
        checkCUDNN(cudnnGetConvolutionBackwardFilterAlgorithm(cudnnHandle, srcTensorDesc, ddstTensorDesc, convDesc,
                                                              dweightDesc, CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0,
                                                              &bf_algo));

        /** Get workspace size */
        workspace_size = get_workspace_size();

        if (workspace_size > 0) {
          checkCudaErrors(cudaMalloc(&workspace, workspace_size));
        }

        if (has_bias) {
          checkCUDNN(cudnnCreateTensorDescriptor(&biasDesc));
          checkCUDNN(cudnnCreateTensorDescriptor(&dbiasDesc));
          checkCUDNN(cudnnSetTensor4dDescriptor(biasDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, out_channels, 1, 1));
          checkCUDNN(cudnnSetTensor4dDescriptor(dbiasDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, out_channels, 1, 1));
        }

        initialized_gpu = true;
      }

      ~Conv_params() {
        if (initialized_gpu) {
          if (workspace) {
            checkCudaErrors(cudaFree(workspace));
          }

          //checkCudaErrors(cublasDestroy(cublasHandle));
          //checkCUDNN(cudnnDestroy(cudnnHandle));
          checkCUDNN(cudnnDestroyTensorDescriptor(srcTensorDesc));
          checkCUDNN(cudnnDestroyTensorDescriptor(dsrcTensorDesc));
          checkCUDNN(cudnnDestroyTensorDescriptor(dstTensorDesc));
          checkCUDNN(cudnnDestroyTensorDescriptor(ddstTensorDesc));
          checkCUDNN(cudnnDestroyTensorDescriptor(biasDesc));
          checkCUDNN(cudnnDestroyTensorDescriptor(dbiasDesc));
          checkCUDNN(cudnnDestroyFilterDescriptor(weightDesc));
          checkCUDNN(cudnnDestroyFilterDescriptor(dweightDesc));
          checkCUDNN(cudnnDestroyConvolutionDescriptor(convDesc));
        }
      }

      size_t get_workspace_size() {
        size_t m = 0;
        size_t s = 0;
        checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle, srcTensorDesc, weightDesc, convDesc,
                                                           dstTensorDesc, fw_algo, &s));
        if (s > m) {
          m = s;
        }

        checkCUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnnHandle, srcTensorDesc, ddstTensorDesc, convDesc,
                                                                  dweightDesc, bf_algo, &s));
        if (s > m) {
          m = s;
        }

        checkCUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(cudnnHandle, weightDesc, ddstTensorDesc, convDesc,
                                                                dsrcTensorDesc, bd_algo, &s));
        if (s > m) {
          m = s;
        }

        return m;
      }
#endif
   private:
#ifdef USE_CUDNN
    bool initialized_gpu = false;
#endif

    };
  }  // namespace core
}  // namespace simpleCNN

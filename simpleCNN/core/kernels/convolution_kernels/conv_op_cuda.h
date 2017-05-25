//
// Created by hacht on 5/20/17.
//

#pragma once

#include "../../../util/simple_error.h"
#include "../../params/conv_params.h"
#include "conv_op_openblas.h"

namespace simpleCNN {
  class ConvCudaForwardOp : public core::OpKernel {
   public:
    explicit ConvCudaForwardOp(const core::OpKernelConstruction& context) : core::OpKernel(context) {
#ifdef USE_CUDNN
      core::Conv_params* conv_params_ptr = static_cast<core::Conv_params*>(core::OpKernel::params_);
      const auto& params                 = conv_params_ptr->conv();

      /** Input */
      checkCUDNN(cudnnCreateTensorDescriptor(&srcTensorDesc));
      checkCUDNN(cudnnSetTensor4dDescriptor(srcTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, params.batch_size,
                                            params.in_channels, params.input_height, params.input_width));

      /** Output */
      checkCUDNN(cudnnCreateTensorDescriptor(&dstTensorDesc));
      checkCUDNN(cudnnSetTensor4dDescriptor(dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, params.batch_size,
                                            params.out_channels, params.output_height, params.output_width));

      /** Weights */
      checkCUDNN(cudnnCreateFilterDescriptor(&weightDesc));
      checkCUDNN(cudnnSetFilter4dDescriptor(weightDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, params.out_channels,
                                            params.in_channels, params.filter_height, params.filter_width));

      /** Convolution specification */
      checkCUDNN(cudnnCreateConvolutionDescriptor(&convDesc));
      checkCUDNN(cudnnSetConvolution2dDescriptor(convDesc, params.padding, params.padding, params.vertical_stride,
                                                 params.horizontal_stride, 1, 1, CUDNN_CROSS_CORRELATION));

      /** Forward prop specification */
      checkCUDNN(cudnnGetConvolutionForwardAlgorithm(cudnn_handle(), srcTensorDesc, weightDesc, convDesc, dstTensorDesc,
                                                     CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &fw_algo));

      workspace_size = get_workspace_size();
      if (workspace_size > 0) {
        checkCudaErrors(cudaMalloc(&workspace, workspace_size));
      }

      checkCudaErrors(cudaMalloc((void**)&input_gpu, sizeof(float_t) * params.batch_size * params.in_channels * params.input_height * params.input_width));
      checkCudaErrors(cudaMalloc((void**)&weight_gpu, sizeof(float_t) * params.out_channels * params.in_channels * params.filter_height * params.filter_width));
      checkCudaErrors(cudaMalloc((void**)&output_gpu, sizeof(float_t) * params.batch_size * params.out_channels * params.output_height * params.output_width));

      /** Bias */
      if (params.has_bias) {
        checkCudaErrors(cudaMalloc((void**)&bias_gpu, sizeof(float_t) * params.out_channels));
        checkCUDNN(cudnnCreateTensorDescriptor(&biasDesc));
        checkCUDNN(cudnnSetTensor4dDescriptor(biasDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, params.out_channels, 1, 1));
      }
#endif
    }

    ~ConvCudaForwardOp() {
#ifdef USE_CUDNN
      if (workspace) {
        checkCudaErrors(cudaFree(workspace));
        workspace_size = 0;
      }

      if (bias_gpu) {
        cuda_free(bias_gpu);
        checkCUDNN(cudnnDestroyTensorDescriptor(biasDesc));
      }

      cuda_free(input_gpu);
      cuda_free(weight_gpu);
      cuda_free(output_gpu);

      checkCUDNN(cudnnDestroyTensorDescriptor(srcTensorDesc));
      checkCUDNN(cudnnDestroyTensorDescriptor(dstTensorDesc));
      checkCUDNN(cudnnDestroyFilterDescriptor(weightDesc));
      checkCUDNN(cudnnDestroyConvolutionDescriptor(convDesc));
#endif
    }

    void compute(const core::OpKernelContext& context) override {
#ifdef USE_CUDNN
      core::Conv_params* conv_params_ptr = static_cast<core::Conv_params*>(core::OpKernel::params_);
      const auto& params                 = conv_params_ptr->conv();

      const tensor_t& in_data = context.input(0);
      const tensor_t& weight  = context.input(1);
      const tensor_t& bias    = context.input(2);
      tensor_t& out_data      = context.output(0);

      /** Push to device memory */
      cuda_push_array(input_gpu, &(*in_data.host_begin()), in_data.size());
      cuda_push_array(weight_gpu, &(*weight.host_begin()), weight.size());

      /** Forward propagate */
      checkCUDNN(cudnnConvolutionForward(cudnn_handle(), &alpha, srcTensorDesc, input_gpu, weightDesc, weight_gpu,
                                         convDesc, fw_algo, workspace, workspace_size, &beta, dstTensorDesc,
                                         output_gpu));

      /** Add bias */
      if (params.has_bias) {
        cuda_push_array(bias_gpu, &(*bias.host_begin()), bias.size());
        checkCUDNN(cudnnAddTensor(cudnn_handle(), &alpha, biasDesc, bias_gpu, &alpha, dstTensorDesc, output_gpu));
      }

      /** Pull from device memory */
      checkCudaErrors(cudaDeviceSynchronize());
      cuda_pull_array(output_gpu, &(*out_data.host_begin()), out_data.size());
#else
      throw simple_error("Not build with gpu support");
#endif
    }

   private:
#ifdef USE_CUDNN
    float_t* input_gpu = nullptr;
    float_t* output_gpu = nullptr;
    float_t* weight_gpu = nullptr;
    float_t* bias_gpu = nullptr;

    float_t alpha = 1.0f;
    float_t beta = 0.0f;


    cudnnTensorDescriptor_t srcTensorDesc;
    cudnnTensorDescriptor_t dstTensorDesc;
    cudnnTensorDescriptor_t biasDesc;
    cudnnFilterDescriptor_t weightDesc;
    cudnnConvolutionDescriptor_t convDesc;
    cudnnConvolutionFwdAlgo_t fw_algo;

    size_t workspace_size = 0;
    void* workspace       = nullptr;

    size_t get_workspace_size() {
      size_t s = 0;
      checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle(), srcTensorDesc, weightDesc, convDesc,
                                                         dstTensorDesc, fw_algo, &s));
      return s;
    }
#endif
  };

  class ConvCudaBackwardOp : public core::OpKernel {
   public:
    explicit ConvCudaBackwardOp(const core::OpKernelConstruction& context) : core::OpKernel(context) {
#ifdef USE_CUDNN
      core::Conv_params* conv_params_ptr = static_cast<core::Conv_params*>(core::OpKernel::params_);
      const auto& params                 = conv_params_ptr->conv();

      /** Input */
      checkCUDNN(cudnnCreateTensorDescriptor(&srcTensorDesc));
      checkCUDNN(cudnnSetTensor4dDescriptor(srcTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, params.batch_size,
                                            params.in_channels, params.input_height, params.input_width));

      /** Backpropr input gradient */
      checkCUDNN(cudnnCreateTensorDescriptor(&dsrcTensorDesc));
      checkCUDNN(cudnnSetTensor4dDescriptor(dsrcTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, params.batch_size,
                                            params.in_channels, params.input_height, params.input_width));

      /** Output */
      checkCUDNN(cudnnCreateTensorDescriptor(&dstTensorDesc));
      checkCUDNN(cudnnSetTensor4dDescriptor(dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, params.batch_size,
                                            params.out_channels, params.output_height, params.output_width));

      /** Backprop output gradient */
      checkCUDNN(cudnnCreateTensorDescriptor(&ddstTensorDesc));
      checkCUDNN(cudnnSetTensor4dDescriptor(ddstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, params.batch_size,
                                            params.out_channels, params.output_height, params.output_width));

      /** Weights */
      checkCUDNN(cudnnCreateFilterDescriptor(&weightDesc));
      checkCUDNN(cudnnSetFilter4dDescriptor(weightDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, params.out_channels,
                                            params.in_channels, params.filter_height, params.filter_width));

      /** dW */
      checkCUDNN(cudnnCreateFilterDescriptor(&dweightDesc));
      checkCUDNN(cudnnSetFilter4dDescriptor(dweightDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, params.out_channels,
                                            params.in_channels, params.filter_height, params.filter_width));

      /** Convolution specification */
      checkCUDNN(cudnnCreateConvolutionDescriptor(&convDesc));
      checkCUDNN(cudnnSetConvolution2dDescriptor_v5(convDesc, params.padding, params.padding, params.vertical_stride,
                                                 params.horizontal_stride, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

      /** Backward data specification */
      checkCUDNN(cudnnGetConvolutionBackwardDataAlgorithm(cudnn_handle(), weightDesc, ddstTensorDesc, convDesc,
                                                          dsrcTensorDesc, CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0,
                                                          &bd_algo));

      /** Backward filter specification */
      checkCUDNN(cudnnGetConvolutionBackwardFilterAlgorithm(cudnn_handle(), srcTensorDesc, ddstTensorDesc, convDesc,
                                                            dweightDesc, CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0,
                                                            &bf_algo));

      /** Get workspace size */
      workspace_size = get_workspace_size();

      if (workspace_size > 0) {
        checkCudaErrors(cudaMalloc(&workspace, workspace_size));
      }

      checkCudaErrors(cudaMalloc((void**)&prev_in_gpu, sizeof(float_t) * params.batch_size * params.in_channels * params.input_height * params.input_width));
      checkCudaErrors(cudaMalloc((void**)&weight_gpu, sizeof(float_t) * params.out_channels * params.in_channels * params.filter_height * params.filter_width));
      checkCudaErrors(cudaMalloc((void**)&dW_gpu, sizeof(float_t) * params.out_channels * params.in_channels * params.filter_height * params.filter_width));
      checkCudaErrors(cudaMalloc((void**)&curr_delta_gpu, sizeof(float_t) * params.batch_size * params.out_channels * params.output_height * params.output_width));
      checkCudaErrors(cudaMalloc((void**)&prev_delta_gpu, sizeof(float_t) * params.batch_size * params.in_channels * params.input_height * params.input_width));

      if (params.has_bias) {
        checkCudaErrors(cudaMalloc((void**)&db_gpu, sizeof(float_t) * params.out_channels));
        checkCUDNN(cudnnCreateTensorDescriptor(&dbiasDesc));
        checkCUDNN(
          cudnnSetTensor4dDescriptor(dbiasDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, params.out_channels, 1, 1));
      }
#endif
    }

    ~ConvCudaBackwardOp() {
#ifdef USE_CUDNN
      if (workspace) {
        checkCudaErrors(cudaFree(workspace));
        workspace_size = 0;
      }

      if (db_gpu) {
        cuda_free(db_gpu);
        checkCUDNN(cudnnDestroyTensorDescriptor(dbiasDesc));
      }

      cuda_free(prev_in_gpu);
      cuda_free(weight_gpu);
      cuda_free(dW_gpu);
      cuda_free(curr_delta_gpu);
      cuda_free(prev_delta_gpu);

      checkCUDNN(cudnnDestroyTensorDescriptor(srcTensorDesc));
      checkCUDNN(cudnnDestroyTensorDescriptor(dsrcTensorDesc));
      checkCUDNN(cudnnDestroyTensorDescriptor(dstTensorDesc));
      checkCUDNN(cudnnDestroyTensorDescriptor(ddstTensorDesc));
      checkCUDNN(cudnnDestroyFilterDescriptor(weightDesc));
      checkCUDNN(cudnnDestroyFilterDescriptor(dweightDesc));
      checkCUDNN(cudnnDestroyConvolutionDescriptor(convDesc));
#endif
    }

    void compute(const core::OpKernelContext& context) override {
#ifdef USE_CUDNN
      core::Conv_params* conv_params_ptr = static_cast<core::Conv_params*>(core::OpKernel::params_);
      const auto& params                 = conv_params_ptr->conv();

      const tensor_t& prev_in = context.input(0);
      const tensor_t& weight  = context.input(1);
      tensor_t& dW            = context.input_grad(1);
      tensor_t& db            = context.input_grad(2);
      tensor_t& prev_delta    = context.input_grad(0);
      tensor_t& curr_delta    = context.output_grad(0);

      /** Psuh to device memory */
      cuda_push_array(prev_in_gpu, &(*prev_in.host_begin()), prev_in.size());
      cuda_push_array(weight_gpu, &(*weight.host_begin()), weight.size());
      cuda_push_array(curr_delta_gpu, &(*curr_delta.host_begin()), curr_delta.size());


      /** Backward propagate */
      checkCUDNN(cudnnConvolutionBackwardFilter(cudnn_handle(), &alpha, srcTensorDesc, prev_in_gpu, ddstTensorDesc,
                                                curr_delta_gpu, convDesc, bf_algo, workspace, workspace_size, &beta,
                                                dweightDesc, dW_gpu));

      //float_t scale = float_t(1) / static_cast<float_t>(params.batch_size);
      //checkCudaErrors(cublasSscal(cublas_handle(), dW.size(), &scale, dW_gpu, 1));

      checkCUDNN(cudnnConvolutionBackwardData(cudnn_handle(), &alpha, weightDesc, weight_gpu, ddstTensorDesc,
                                              curr_delta_gpu, convDesc, bd_algo, workspace, workspace_size, &beta,
                                              dsrcTensorDesc, prev_delta_gpu));

      /** Backprop bias */
      if (params.has_bias) {
        checkCUDNN(cudnnConvolutionBackwardBias(cudnn_handle(), &alpha, ddstTensorDesc, curr_delta_gpu, &beta, dbiasDesc, db_gpu));
        //checkCudaErrors(cublasSscal(cublas_handle(), db.size(), &scale, db_gpu, 1));

        checkCudaErrors(cudaDeviceSynchronize());
        cuda_pull_array(db_gpu, &(*db.host_begin()), db.size());
      }

      /** Pull from device memory */
      checkCudaErrors(cudaDeviceSynchronize());
      cuda_pull_array(prev_delta_gpu, &(*prev_delta.host_begin()), prev_delta.size());
      cuda_pull_array(dW_gpu, &(*dW.host_begin()), dW.size());
#else
      throw simple_error("Running on gpu when not built with gpu support");
#endif
    }

   private:
#ifdef USE_CUDNN
    float_t* prev_in_gpu    = nullptr;
    float_t* weight_gpu     = nullptr;
    float_t* dW_gpu         = nullptr;
    float_t* db_gpu         = nullptr;
    float_t* prev_delta_gpu = nullptr;
    float_t* curr_delta_gpu = nullptr;

    float_t alpha = 1.0f;
    float_t beta = 0.0f;


  cudnnTensorDescriptor_t srcTensorDesc, dstTensorDesc;
    cudnnTensorDescriptor_t dsrcTensorDesc, ddstTensorDesc, dbiasDesc;
    cudnnFilterDescriptor_t weightDesc, dweightDesc;
    cudnnConvolutionDescriptor_t convDesc;

    cudnnConvolutionBwdFilterAlgo_t bf_algo;
    cudnnConvolutionBwdDataAlgo_t bd_algo;

    size_t workspace_size = 0;
    void* workspace       = nullptr;

    size_t get_workspace_size() {
      size_t m = 0;
      size_t s = 0;

      checkCUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnn_handle(), srcTensorDesc, ddstTensorDesc, convDesc,
                                                                dweightDesc, bf_algo, &s));
      if (s > m) {
        m = s;
      }

      checkCUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(cudnn_handle(), weightDesc, ddstTensorDesc, convDesc,
                                                              dsrcTensorDesc, bd_algo, &s));
      if (s > m) {
        m = s;
      }

      return m;
    }
#endif
  };
}  // namespace simpleCNN
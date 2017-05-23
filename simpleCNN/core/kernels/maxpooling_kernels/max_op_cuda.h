//
// Created by hacht on 5/21/17.
//

#pragma once

#include "../../params/max_params.h"
#include "max_op_internal.h"

namespace simpleCNN {
  class MaxpoolingCudaForwardOp : public core::OpKernel {
   public:
    explicit MaxpoolingCudaForwardOp(const core::OpKernelConstruction& context) : core::OpKernel(context) {
#ifdef USE_CUDNN
      core::Maxpooling_params* maxpooling_params_ptr = static_cast<core::Maxpooling_params*>(core::OpKernel::params_);
      const auto& params                             = maxpooling_params_ptr->maxpool();

      checkCUDNN(cudnnCreateTensorDescriptor(&srcTensorDesc));
      checkCUDNN(cudnnSetTensor4dDescriptor(srcTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, params.batch_size,
                                            params.in_channels, params.input_height, params.input_width));

      checkCUDNN(cudnnCreateTensorDescriptor(&dstTensorDesc));
      checkCUDNN(cudnnSetTensor4dDescriptor(dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, params.batch_size,
                                            params.out_channels, params.output_height, params.output_width));

      checkCUDNN(cudnnCreatePoolingDescriptor(&poolDesc));
      checkCUDNN(cudnnSetPooling2dDescriptor(poolDesc, CUDNN_POOLING_MAX, CUDNN_PROPAGATE_NAN, params.pooling_size_y,
                                             params.pooling_size_x, 0, 0, params.stride_y, params.stride_x));
#endif
    }

    ~MaxpoolingCudaForwardOp() {
#ifdef USE_CUDNN
      checkCUDNN(cudnnDestroyTensorDescriptor(srcTensorDesc));
      checkCUDNN(cudnnDestroyTensorDescriptor(dstTensorDesc));
      checkCUDNN(cudnnDestroyPoolingDescriptor(poolDesc));
#endif
    }

    void compute(const core::OpKernelContext& context) override {
#ifdef USE_CUDNN
      core::Maxpooling_params* maxpooling_params_ptr = static_cast<core::Maxpooling_params*>(core::OpKernel::params_);
      const auto& params                             = maxpooling_params_ptr->maxpool();

      const tensor_t& in_data = context.input(0);
      tensor_t& out_data      = context.output(0);

      /** Initalize device memory */
      float_t* in_data_gpu  = cuda_make_array(&(*in_data.host_begin()), in_data.size());
      float_t* out_data_gpu = cuda_make_array(&(*out_data.host_begin()), out_data.size());

      /** Forward propagate */
      float_t one = 1;
      checkCUDNN(cudnnPoolingForward(cudnn_handle(), poolDesc, &one, srcTensorDesc, in_data_gpu, &one, dstTensorDesc,
                                     out_data_gpu));

      /** Pull result from device */
      checkCudaErrors(cudaDeviceSynchronize());
      cuda_pull_array(out_data_gpu, &(*out_data.host_begin()), out_data.size());

      /** Release allocated gpu memory */
      cuda_free(in_data_gpu);
      cuda_free(out_data_gpu);
#else
      throw simple_error("Running on gpu when not built with gpu support");
#endif
    }

   private:
#ifdef USE_CUDNN
    cudnnTensorDescriptor_t srcTensorDesc;
    cudnnTensorDescriptor_t dstTensorDesc;
    cudnnPoolingDescriptor_t poolDesc;
#endif
  };

  class MaxpoolingCudaBackwardOp : public core::OpKernel {
   public:
    explicit MaxpoolingCudaBackwardOp(const core::OpKernelConstruction& context) : core::OpKernel(context) {
#ifdef USE_CUDNN
      core::Maxpooling_params* maxpooling_params_ptr = static_cast<core::Maxpooling_params*>(core::OpKernel::params_);
      const auto& params                             = maxpooling_params_ptr->maxpool();

      checkCUDNN(cudnnCreateTensorDescriptor(&srcTensorDesc));
      checkCUDNN(cudnnSetTensor4dDescriptor(srcTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, params.batch_size,
                                            params.in_channels, params.input_height, params.input_width));
      checkCUDNN(cudnnCreateTensorDescriptor(&dsrcTensorDesc));
      checkCUDNN(cudnnSetTensor4dDescriptor(dsrcTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, params.batch_size,
                                            params.in_channels, params.input_height, params.input_width));

      checkCUDNN(cudnnCreateTensorDescriptor(&dstTensorDesc));
      checkCUDNN(cudnnSetTensor4dDescriptor(dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, params.batch_size,
                                            params.out_channels, params.output_height, params.output_width));

      checkCUDNN(cudnnCreateTensorDescriptor(&ddstTensorDesc));
      checkCUDNN(cudnnSetTensor4dDescriptor(ddstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, params.batch_size,
                                            params.out_channels, params.output_height, params.output_width));

      checkCUDNN(cudnnCreatePoolingDescriptor(&poolDesc));
      checkCUDNN(cudnnSetPooling2dDescriptor(poolDesc, CUDNN_POOLING_MAX, CUDNN_PROPAGATE_NAN, params.pooling_size_y,
                                             params.pooling_size_x, 0, 0, params.stride_y, params.stride_x));
#endif
    }

    ~MaxpoolingCudaBackwardOp() {
#ifdef USE_CUDNN
      checkCUDNN(cudnnDestroyTensorDescriptor(srcTensorDesc));
      checkCUDNN(cudnnDestroyTensorDescriptor(dsrcTensorDesc));
      checkCUDNN(cudnnDestroyTensorDescriptor(dstTensorDesc));
      checkCUDNN(cudnnDestroyTensorDescriptor(ddstTensorDesc));
      checkCUDNN(cudnnDestroyPoolingDescriptor(poolDesc));
#endif
    }

    void compute(const core::OpKernelContext& context) override {
#ifdef USE_CUDNN
      core::Maxpooling_params* maxpooling_params_ptr = static_cast<core::Maxpooling_params*>(core::OpKernel::params_);
      const auto& params                             = maxpooling_params_ptr->maxpool();

      const tensor_t& in_data    = context.input(0);
      const tensor_t& out_data   = context.output(0);
      const tensor_t& curr_delta = context.output_grad(0);
      tensor_t& prev_delta       = context.input_grad(0);

      /** Initalize device memory */
      float_t* in_data_gpu    = cuda_make_array(&*(in_data.host_begin()), in_data.size());
      float_t* out_data_gpu   = cuda_make_array(&(*out_data.host_begin()), out_data.size());
      float_t* curr_delta_gpu = cuda_make_array(&(*curr_delta.host_begin()), curr_delta.size());
      float_t* prev_delta_gpu = cuda_make_array(&(*prev_delta.host_begin()), prev_delta.size());

      /** Backward propagate */
      float_t one = 1;
      checkCUDNN(cudnnPoolingBackward(cudnn_handle(), poolDesc, &one, dstTensorDesc, out_data_gpu, ddstTensorDesc,
                                      curr_delta_gpu, srcTensorDesc, in_data_gpu, &one, dsrcTensorDesc,
                                      prev_delta_gpu));

      /** Pull result from device */
      checkCudaErrors(cudaDeviceSynchronize());
      cuda_pull_array(prev_delta_gpu, &(*prev_delta.host_begin()), prev_delta.size());

      /** Release allocated gpu memory */
      cuda_free(in_data_gpu);
      cuda_free(out_data_gpu);
      cuda_free(curr_delta_gpu);
      cuda_free(prev_delta_gpu);
#else
      throw simple_error("Running on gpu when not built with gpu support");
#endif
    }

   private:
#ifdef USE_CUDNN
    cudnnTensorDescriptor_t srcTensorDesc;
    cudnnTensorDescriptor_t dstTensorDesc;
    cudnnTensorDescriptor_t dsrcTensorDesc;
    cudnnTensorDescriptor_t ddstTensorDesc;
    cudnnPoolingDescriptor_t poolDesc;
#endif
  };
}  // namespace simpleCNN
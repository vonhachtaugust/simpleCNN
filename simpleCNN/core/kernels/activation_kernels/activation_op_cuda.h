//
// Created by hacht on 5/22/17.
//

#pragma once

#include "../../framework/op_kernel.h"
#include "../../params/activation_params.h"

#ifdef USE_CUDNN
#include "../../../../third_party/cudnn/include/cudnn.h"
#include "../../../util/cuda_utils.h"
#endif

namespace simpleCNN {

  class ActivationCudaForwardOp : public core::OpKernel {
   public:
    explicit ActivationCudaForwardOp(const core::OpKernelConstruction& context) : core::OpKernel(context) {
#ifdef USE_CUDNN
      core::Activation_params* active_params_ptr = static_cast<core::Activation_params*>(core::OpKernel::params_);
      const auto& params                         = active_params_ptr->activation_params();

      if (product(params.shape) == 0) {
        throw simple_error("Activation layer shape not initialized");
      }

      checkCUDNN(cudnnCreateTensorDescriptor(&srcTensorDesc));
      checkCUDNN(cudnnSetTensor4dDescriptor(srcTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, params.shape[0],
                                            params.shape[1], params.shape[2], params.shape[3]));

      checkCUDNN(cudnnCreateTensorDescriptor(&dstTensorDesc));
      checkCUDNN(cudnnSetTensor4dDescriptor(dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, params.shape[0],
                                            params.shape[1], params.shape[2], params.shape[3]));

      checkCudaErrors(cudaMalloc((void**)&input_gpu, sizeof(float_t) * product(params.shape)));
      checkCudaErrors(cudaMalloc((void**)&output_gpu, sizeof(float_t) * product(params.shape)));

      if (params.activation_function == core::activation_t::relu) {
        checkCUDNN(cudnnCreateActivationDescriptor(&Activation));
        checkCUDNN(cudnnSetActivationDescriptor(Activation, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0.0));
      } else {
        throw simple_error("No activation initialized");
      }
#endif
    }

    ~ActivationCudaForwardOp() {
#ifdef USE_CUDNN
      cuda_free(input_gpu);
      cuda_free(output_gpu);

      checkCUDNN(cudnnDestroyActivationDescriptor(Activation));
      checkCUDNN(cudnnDestroyTensorDescriptor(srcTensorDesc));
      checkCUDNN(cudnnDestroyTensorDescriptor(dstTensorDesc));
#endif
    }

    void compute(const core::OpKernelContext& context) override {
#ifdef USE_CUDNN
      const tensor_t& input = context.input(0);
      tensor_t& output      = context.output(0);

      /** Push to device memory */
      cuda_push_array(input_gpu, &(*input.host_begin()), input.size());

      /** Forward propagate */
      checkCUDNN(cudnnActivationForward(cudnn_handle(), Activation, &alpha, srcTensorDesc, input_gpu, &beta,
                                        dstTensorDesc, output_gpu));

      /** Pull from device memory */
      checkCudaErrors(cudaDeviceSynchronize());
      cuda_pull_array(output_gpu, &(*output.host_begin()), output.size());
#endif
    }

   private:
#ifdef USE_CUDNN
    float_t* input_gpu  = nullptr;
    float_t* output_gpu = nullptr;

    float_t alpha = 1.0f;
    float_t beta  = 0.0f;

    cudnnTensorDescriptor_t srcTensorDesc;
    cudnnTensorDescriptor_t dstTensorDesc;
    cudnnActivationDescriptor_t Activation;
#endif
  };

  class ActivationCudaBackwardOp : public core::OpKernel {
   public:
    explicit ActivationCudaBackwardOp(const core::OpKernelConstruction& context) : core::OpKernel(context) {
#ifdef USE_CUDNN
      core::Activation_params* active_params_ptr = static_cast<core::Activation_params*>(core::OpKernel::params_);
      const auto& params                         = active_params_ptr->activation_params();

      if (params.shape.size() == 0) {
        throw simple_error("Activation layer shape not initialized");
      }

      checkCUDNN(cudnnCreateTensorDescriptor(&srcTensorDesc));
      checkCUDNN(cudnnSetTensor4dDescriptor(srcTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, params.shape[0],
                                            params.shape[1], params.shape[2], params.shape[3]));
      checkCUDNN(cudnnCreateTensorDescriptor(&dsrcTensorDesc));
      checkCUDNN(cudnnSetTensor4dDescriptor(dsrcTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, params.shape[0],
                                            params.shape[1], params.shape[2], params.shape[3]));

      checkCUDNN(cudnnCreateTensorDescriptor(&dstTensorDesc));
      checkCUDNN(cudnnSetTensor4dDescriptor(dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, params.shape[0],
                                            params.shape[1], params.shape[2], params.shape[3]));

      checkCUDNN(cudnnCreateTensorDescriptor(&ddstTensorDesc));
      checkCUDNN(cudnnSetTensor4dDescriptor(ddstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, params.shape[0],
                                            params.shape[1], params.shape[2], params.shape[3]));

      checkCudaErrors(cudaMalloc((void**)&input_gpu, sizeof(float_t) * product(params.shape)));
      checkCudaErrors(cudaMalloc((void**)&output_gpu, sizeof(float_t) * product(params.shape)));
      checkCudaErrors(cudaMalloc((void**)&curr_delta_gpu, sizeof(float_t) * product(params.shape)));
      checkCudaErrors(cudaMalloc((void**)&prev_delta_gpu, sizeof(float_t) * product(params.shape)));

      if (params.activation_function == core::activation_t::relu) {
        checkCUDNN(cudnnCreateActivationDescriptor(&Activation));
        checkCUDNN(cudnnSetActivationDescriptor(Activation, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0.0));
      } else {
        throw simple_error("No activation initialized");
      }
#endif
    }

    ~ActivationCudaBackwardOp() {
#ifdef USE_CUDNN
      cuda_free(input_gpu);
      cuda_free(output_gpu);
      cuda_free(curr_delta_gpu);
      cuda_free(prev_delta_gpu);

      checkCUDNN(cudnnDestroyTensorDescriptor(srcTensorDesc));
      checkCUDNN(cudnnDestroyTensorDescriptor(dstTensorDesc));
      checkCUDNN(cudnnDestroyTensorDescriptor(dsrcTensorDesc));
      checkCUDNN(cudnnDestroyTensorDescriptor(ddstTensorDesc));
      checkCUDNN(cudnnDestroyActivationDescriptor(Activation));
#endif
    }

    void compute(const core::OpKernelContext& context) override {
#ifdef USE_CUDNN
      const tensor_t& input      = context.input(0);
      const tensor_t& output     = context.output(0);
      const tensor_t& curr_delta = context.output_grad(0);
      tensor_t& prev_delta       = context.input_grad(0);

      /** Push to device memory */
      cuda_push_array(input_gpu, &(*input.host_begin()), input.size());
      cuda_push_array(output_gpu, &(*output.host_begin()), output.size());
      cuda_push_array(curr_delta_gpu, &(*curr_delta.host_begin()), curr_delta.size());

      /** Backward propagate */
      checkCUDNN(cudnnActivationBackward(cudnn_handle(), Activation, &alpha, dstTensorDesc, output_gpu, ddstTensorDesc,
                                         curr_delta_gpu, srcTensorDesc, input_gpu, &beta, dsrcTensorDesc,
                                         prev_delta_gpu));

      /**  Pull from device memory */
      checkCudaErrors(cudaDeviceSynchronize());
      cuda_pull_array(prev_delta_gpu, &(*prev_delta.host_begin()), prev_delta.size());
#endif
    }

   private:
#ifdef USE_CUDNN
    float_t* input_gpu      = nullptr;
    float_t* output_gpu     = nullptr;
    float_t* curr_delta_gpu = nullptr;
    float_t* prev_delta_gpu = nullptr;

    float_t alpha = 1.0f;
    float_t beta  = 0.0f;

    cudnnTensorDescriptor_t srcTensorDesc, dstTensorDesc;
    cudnnTensorDescriptor_t dsrcTensorDesc, ddstTensorDesc;
    cudnnActivationDescriptor_t Activation;
#endif
  };
}  // namespace simpleCNN
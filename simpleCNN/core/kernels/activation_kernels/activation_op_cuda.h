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
    const auto& params = active_params_ptr->activation_params();

    if (product(params.shape) == 0) {
      throw simple_error("Activation layer shape not initialized");
    }

    checkCUDNN(cudnnCreateTensorDescriptor(&srcTensorDesc));
    checkCUDNN(cudnnSetTensor4dDescriptor(srcTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, params.shape[0], params.shape[1],
                                          params.shape[2], params.shape[3]));

    checkCUDNN(cudnnCreateTensorDescriptor(&dstTensorDesc));
    checkCUDNN(cudnnSetTensor4dDescriptor(dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, params.shape[0], params.shape[1],
                                          params.shape[2], params.shape[3]));


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
    checkCUDNN(cudnnDestroyActivationDescriptor(Activation));
    checkCUDNN(cudnnDestroyTensorDescriptor(srcTensorDesc));
    checkCUDNN(cudnnDestroyTensorDescriptor(dstTensorDesc));
#endif
  }


  void compute(const core::OpKernelContext& context) override {
#ifdef USE_CUDNN
    const tensor_t& input = context.input(0);
    tensor_t& output      = context.output(0);

    /** Initialize device memory */
    float_t* input_gpu    = cuda_make_array(&(*input.host_begin()), input.size());
    float_t* output_gpu = cuda_make_array(&(*output.host_begin()), output.size());

    /** Forward propagate */
    float_t one = 1;
    checkCUDNN(cudnnActivationForward(cudnn_handle(), Activation, &one,
                                      srcTensorDesc, input_gpu, &one,
                                      dstTensorDesc, output_gpu));

    /** Pull result from device */
    checkCudaErrors(cudaDeviceSynchronize());
    cuda_pull_array(output_gpu, &(*output.host_begin()), output.size());

    /** Release allocated gpu memory */
    cuda_free(input_gpu);
    cuda_free(output_gpu);
#endif
  }

 private:
#ifdef USE_CUDNN
  cudnnTensorDescriptor_t srcTensorDesc;
  cudnnTensorDescriptor_t dstTensorDesc;
  cudnnActivationDescriptor_t Activation;
#endif

};

class ActivationCudaBackwardOp : public core::OpKernel {
 public:
  explicit ActivationCudaBackwardOp(const core::OpKernelConstruction &context) : core::OpKernel(context) {
#ifdef USE_CUDNN
    core::Activation_params* active_params_ptr = static_cast<core::Activation_params*>(core::OpKernel::params_);
    const auto& params = active_params_ptr->activation_params();

    if (params.shape.size() == 0) {
      throw simple_error("Activation layer shape not initialized");
    }

    checkCUDNN(cudnnCreateTensorDescriptor(&srcTensorDesc));
    checkCUDNN(cudnnSetTensor4dDescriptor(srcTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, params.shape[0], params.shape[1],
                                          params.shape[2], params.shape[3]));
    checkCUDNN(cudnnCreateTensorDescriptor(&dsrcTensorDesc));
    checkCUDNN(cudnnSetTensor4dDescriptor(dsrcTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, params.shape[0], params.shape[1],
                                          params.shape[2], params.shape[3]));

    checkCUDNN(cudnnCreateTensorDescriptor(&dstTensorDesc));
    checkCUDNN(cudnnSetTensor4dDescriptor(dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, params.shape[0], params.shape[1],
                                          params.shape[2], params.shape[3]));

    checkCUDNN(cudnnCreateTensorDescriptor(&ddstTensorDesc));
    checkCUDNN(cudnnSetTensor4dDescriptor(ddstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, params.shape[0], params.shape[1],
                                          params.shape[2], params.shape[3]));


    if (params.activation_function == core::activation_t::relu) {
      checkCUDNN(cudnnCreateActivationDescriptor(&Activation));
      checkCUDNN(cudnnSetActivationDescriptor(Activation, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0.0));
    } else {
      throw simple_error("No activation initialized");
    }
#endif
  }

  void compute(const core::OpKernelContext &context) override {
#ifdef USE_CUDNN
    const tensor_t &input = context.input(0);
    const tensor_t &output = context.output(0);
    const tensor_t &curr_delta    = context.output_grad(0);
    tensor_t &prev_delta    = context.input_grad(0);

    /** Initialize device memory */
    float_t* input_gpu     = cuda_make_array(&(*input.host_begin()), input.size());
    float_t* output_gpu  = cuda_make_array(&(*output.host_begin()), output.size());
    float_t* curr_delta_gpu = cuda_make_array(&(*curr_delta.host_begin()), curr_delta.size());
    float_t* prev_delta_gpu = cuda_make_array(&(*prev_delta.host_begin()), prev_delta.size());

    /** Backward propagate */
    float_t one = 1;
    checkCUDNN(cudnnActivationBackward(
        cudnn_handle(), Activation, &one, dstTensorDesc,
        output_gpu, ddstTensorDesc, curr_delta_gpu, srcTensorDesc, input_gpu,
        &one, dsrcTensorDesc, prev_delta_gpu));

    /**  Pull result from device */
    checkCudaErrors(cudaDeviceSynchronize());
    cuda_pull_array(prev_delta_gpu, &(*prev_delta.host_begin()), prev_delta.size());

    /** Release allocated gpu memory */
    cuda_free(input_gpu);
    cuda_free(output_gpu);
    cuda_free(curr_delta_gpu);
    cuda_free(prev_delta_gpu);
#endif
  }

 private:
#ifdef USE_CUDNN
  cudnnTensorDescriptor_t srcTensorDesc, dstTensorDesc;
  cudnnTensorDescriptor_t dsrcTensorDesc, ddstTensorDesc;
  cudnnActivationDescriptor_t Activation;
#endif
};
} // namespace simpleCNN
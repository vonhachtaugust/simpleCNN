//
// Created by hacht on 5/20/17.
//

#pragma once

#include "../params/conv_params.h"
#include "conv_op_openblas.h"

namespace simpleCNN {
  class ConvCudaForwardOp : public core::OpKernel {
   public:
    explicit ConvCudaForwardOp(const core::OpKernelConstruction& context) : core::OpKernel(context) {}

    void compute(const core::OpKernelContext& context) override {
#ifdef USE_CUDNN
      core::Conv_params* conv_params_ptr = static_cast<core::Conv_params*>(core::OpKernel::params_);
      const auto& params                 = conv_params_ptr->conv();

      const tensor_t& in_data = context.input(0);
      const tensor_t& weight  = context.input(1);
      const tensor_t& bias    = context.input(2);
      tensor_t& out_data      = context.output(0);

      /** Initalize device memory */
      float_t* in_data_gpu  = cuda_make_array(&(*in_data.host_begin()), in_data.size());
      float_t* weight_gpu   = cuda_make_array(&(*weight.host_begin()), weight.size());
      float_t* out_data_gpu = cuda_make_array(&(*out_data.host_begin()), out_data.size());

      /** Forward propagate */
      float_t one = 1;
      checkCUDNN(cudnnConvolutionForward(params.cudnnHandle, &one, params.srcTensorDesc, in_data_gpu, params.weightDesc,
                                         weight_gpu, params.convDesc, params.fw_algo, params.workspace,
                                         params.workspace_size, &one, params.dstTensorDesc, out_data_gpu));

      /** Add bias */
      if (params.has_bias) {
        float_t* bias_gpu = cuda_make_array(&(*bias.host_begin()), bias.size());

        checkCUDNN(cudnnAddTensor(params.cudnnHandle, &one, params.biasDesc, bias_gpu, &one, params.dstTensorDesc,
                                  out_data_gpu));
        cuda_free(bias_gpu);
      }

      /** Pull result from device to host */
      cuda_pull_array(out_data_gpu, &(*out_data.host_begin()), out_data.size());

      /** Release allocated gpu memory */
      cuda_free(in_data_gpu);
      cuda_free(out_data_gpu);
      cuda_free(weight_gpu);
#endif
    }
  };

  class ConvCudaBackwardOp : public core::OpKernel {
   public:
    explicit ConvCudaBackwardOp(const core::OpKernelConstruction& context) : core::OpKernel(context) {}

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

      /** Initialize device memory */
      float_t* prev_in_gpu    = cuda_make_array(&(*prev_in.host_begin()), prev_in.size());
      float_t* weight_gpu     = cuda_make_array(&(*weight.host_begin()), weight.size());
      float_t* dW_gpu         = cuda_make_array(&(*dW.host_begin()), dW.size());
      float_t* prev_delta_gpu = cuda_make_array(&(*prev_delta.host_begin()), prev_delta.size());
      float_t* curr_delta_gpu = cuda_make_array(&(*curr_delta.host_begin()), curr_delta.size());

      /** Backward propagate */
      float_t one = 1;
      checkCUDNN(cudnnConvolutionBackwardFilter(
        params.cudnnHandle, &one, params.srcTensorDesc, prev_in_gpu, params.ddstTensorDesc, curr_delta_gpu,
        params.convDesc, params.bf_algo, params.workspace, params.workspace_size, &one, params.dweightDesc, dW_gpu));

      float_t alpha = float_t(1) / static_cast<float_t>(params.batch_size);
      checkCudaErrors(cublasSscal(params.cublasHandle,
                                  dW.size(),
                                  &alpha,
                                  dW_gpu,
                                  one));

      checkCUDNN(cudnnConvolutionBackwardData(
        params.cudnnHandle, &one, params.weightDesc, weight_gpu, params.ddstTensorDesc, curr_delta_gpu, params.convDesc,
        params.bd_algo, params.workspace, params.workspace_size, &one, params.dsrcTensorDesc, prev_delta_gpu));

      /** Backprop bias */
      if (params.has_bias) {
        float_t* db_gpu = cuda_make_array(&(*db.host_begin()), db.size());

        checkCUDNN(cudnnConvolutionBackwardBias(params.cudnnHandle, &one, params.dstTensorDesc, curr_delta_gpu, &one,
                                                params.biasDesc, db_gpu));

        checkCudaErrors(cublasSscal(params.cublasHandle,
                                    db.size(),
                                    &alpha,
                                    db_gpu,
                                    one));
        cuda_pull_array(db_gpu, &(*db.host_begin()), db.size());
        cuda_free(db_gpu);
      }

      /** Pull result from device to host */
      cuda_pull_array(prev_delta_gpu, &(*prev_delta.host_begin()), prev_delta.size());
      cuda_pull_array(dW_gpu, &(*dW.host_begin()), dW.size());

      /** Release allocated gpu memory */
      cuda_free(prev_delta_gpu);
      cuda_free(weight_gpu);
      cuda_free(dW_gpu);
      cuda_free(curr_delta_gpu);
#endif
    }
  };
}  // namespace simpleCNN
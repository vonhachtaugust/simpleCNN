//
// Created by hacht on 5/21/17.
//

#pragma once

#include "../framework/op_kernel.h"
#include "../params/con_params.h"

namespace simpleCNN {

  class ConCudaForwardOp : public core::OpKernel {
   public:
    explicit ConCudaForwardOp(const core::OpKernelConstruction& context) : core::OpKernel(context) {}

    void compute(const core::OpKernelContext& context) override {
#ifdef USE_CUDNN
      core::Con_params* connected_params_ptr = static_cast<core::Con_params*>(core::OpKernel::params_);
      const auto& params                     = connected_params_ptr->connected_params();

      const tensor_t& in_data = context.input(0);
      const tensor_t& weight  = context.input(1);
      const tensor_t& bias    = context.input(2);
      tensor_t& out_data      = context.output(0);

      /** Initialize device memory */
      float_t* in_data_gpu  = cuda_make_array(&(*in_data.host_begin()), in_data.size());
      float_t* weight_gpu   = cuda_make_array(&(*weight.host_begin()), weight.size());
      float_t* out_data_gpu = cuda_make_array(&(*out_data.host_begin()), out_data.size());

      /** Forward propagate */
      float_t one = 1;
      checkCudaErrors(cublasSgemm(params.cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, params.out_dim, params.batch_size,
                                  params.in_dim, &one, weight_gpu, params.in_dim, in_data_gpu, params.in_dim, &one,
                                  out_data_gpu, params.out_dim));

      /** Add bias */
      if (params.has_bias) {
        float_t* bias_gpu = cuda_make_array(&(*bias.host_begin()), bias.size());

        checkCudaErrors(cublasSgemm(params.cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, params.out_dim, params.batch_size,
                                    one, &one, bias_gpu, params.out_dim, params.onevec, one, &one, out_data_gpu,
                                    params.out_dim));
        cuda_free(bias_gpu);
      }

      /** Pull result from device */
      cuda_pull_array(out_data_gpu, &(*out_data.host_begin()), out_data.size());

      /** Release allocated gpu mmemory */
      cuda_free(in_data_gpu);
      cuda_free(out_data_gpu);
      cuda_free(weight_gpu);
#endif
    }
  };

  class ConCudaBackwardGradOp : public core::OpKernel {
   public:
    explicit ConCudaBackwardGradOp(const core::OpKernelConstruction& context) : core::OpKernel(context) {}

    void compute(const core::OpKernelContext& context) override {
#ifdef USE_CUDNN
      core::Con_params* connected_params_ptr = static_cast<core::Con_params*>(core::OpKernel::params_);
      const auto& params                     = connected_params_ptr->connected_params();

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
      // Weights
      checkCudaErrors(cublasSgemm(params.cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, params.in_dim, params.out_dim,
                                  params.batch_size, &one, prev_in_gpu, params.in_dim, curr_delta_gpu, params.out_dim,
                                  &one, dW_gpu, params.in_dim));

      // scale due to batch size
      float_t alpha = float_t(1) / static_cast<float_t>(params.batch_size);
      checkCudaErrors(cublasSscal(params.cublasHandle,
                                  dW.size(),
                                  &alpha,
                                  dW_gpu,
                                  one));

      // Data
      checkCudaErrors(cublasSgemm(params.cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, params.in_dim, params.batch_size,
                                  params.out_dim, &one, weight_gpu, params.in_dim, curr_delta_gpu, params.out_dim, &one,
                                  prev_delta_gpu, params.in_dim));

      if (params.has_bias) {
        float_t* db_gpu = cuda_make_array(&(*db.host_begin()), db.size());

        checkCudaErrors(cublasSgemv(params.cublasHandle, CUBLAS_OP_N, params.out_dim, params.batch_size, &one,
                                    curr_delta_gpu, params.out_dim, params.onevec, one, &one, db_gpu, one));

        checkCudaErrors(cublasSscal(params.cublasHandle,
                                    db.size(),
                                    &alpha,
                                    db_gpu,
                                    one));
        cuda_pull_array(db_gpu, &(*db.host_begin()), db.size());
        cuda_free(db_gpu);
      }

      /** Pull result from device */
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
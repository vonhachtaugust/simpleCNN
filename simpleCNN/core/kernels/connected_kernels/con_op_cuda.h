//
// Created by hacht on 5/21/17.
//

#pragma once

#include "../../framework/op_kernel.h"
#include "../../params/con_params.h"

#ifdef USE_CUDNN
#include "../cuda_util_kernels.h"
#endif

namespace simpleCNN {

  class ConCudaForwardOp : public core::OpKernel {
   public:
    explicit ConCudaForwardOp(const core::OpKernelConstruction& context) : core::OpKernel(context) {
#ifdef USE_CUDNN
      core::Con_params* connected_params_ptr = static_cast<core::Con_params*>(core::OpKernel::params_);
      const auto& params                     = connected_params_ptr->connected_params();

      checkCudaErrors(cudaMalloc((void**)&input_gpu, sizeof(float_t) * params.batch_size * params.in_dim));
      checkCudaErrors(cudaMalloc((void**)&weight_gpu, sizeof(float_t) * params.in_dim * params.out_dim));
      checkCudaErrors(cudaMalloc((void**)&output_gpu, sizeof(float_t) * params.batch_size * params.out_dim));

      if (params.has_bias) {
        checkCudaErrors(cudaMalloc((void**)&bias_gpu, sizeof(float_t) * params.out_dim));
        checkCudaErrors(cudaMalloc((void**)&onevec, sizeof(float_t) * params.batch_size));

        tensor_t ones({params.batch_size, 1, 1, 1});
        ones.fill(1.0f);
        cuda_push_array(onevec, &(*ones.host_begin()), ones.size());
      }
#endif
    }

    ~ConCudaForwardOp() {
#ifdef USE_CUDNN
      cuda_free(input_gpu);
      cuda_free(output_gpu);
      cuda_free(weight_gpu);

      if (bias_gpu) {
        cuda_free(onevec);
        cuda_free(bias_gpu);
      }
#endif
    }

    void compute(const core::OpKernelContext& context) override {
#ifdef USE_CUDNN
      core::Con_params* connected_params_ptr = static_cast<core::Con_params*>(core::OpKernel::params_);
      const auto& params                     = connected_params_ptr->connected_params();

      const tensor_t& in_data = context.input(0);
      const tensor_t& weight  = context.input(1);
      const tensor_t& bias    = context.input(2);
      tensor_t& out_data      = context.output(0);

      /** Push to device memory */
      cuda_push_array(input_gpu, &(*in_data.host_begin()), in_data.size());
      cuda_push_array(weight_gpu, &(*weight.host_begin()), weight.size());

      /** Forward propagate */
      checkCudaErrors(cublasSgemm(cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N, params.out_dim, params.batch_size,
                                  params.in_dim, &alpha, weight_gpu, params.in_dim, input_gpu, params.in_dim, &beta,
                                  output_gpu, params.out_dim));

      /** Add bias */
      if (params.has_bias) {
        cuda_push_array(bias_gpu, &(*bias.host_begin()), bias.size());
        checkCudaErrors(cublasSgemm(cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N, params.out_dim, params.batch_size, 1,
                                    &alpha, bias_gpu, params.out_dim, onevec, 1, &alpha, output_gpu, params.out_dim));
      }

      /** Pull from device memory */
      checkCudaErrors(cudaDeviceSynchronize());
      cuda_pull_array(output_gpu, &(*out_data.host_begin()), out_data.size());
#else
      throw simple_error("Running on gpu when not built with gpu support");
#endif
    }

   private:
#ifdef USE_CUDNN
    float_t* input_gpu  = nullptr;
    float_t* output_gpu = nullptr;
    float_t* weight_gpu = nullptr;
    float_t* bias_gpu   = nullptr;

    float_t alpha = 1.0f;
    float_t beta  = 0.0f;

    float_t* onevec = nullptr;
#endif
  };

  class ConCudaBackwardGradOp : public core::OpKernel {
   public:
    explicit ConCudaBackwardGradOp(const core::OpKernelConstruction& context) : core::OpKernel(context) {
#ifdef USE_CUDNN
      core::Con_params* connected_params_ptr = static_cast<core::Con_params*>(core::OpKernel::params_);
      const auto& params                     = connected_params_ptr->connected_params();

      checkCudaErrors(cudaMalloc((void**)&prev_in_gpu, sizeof(float_t) * params.batch_size * params.in_dim));
      checkCudaErrors(cudaMalloc((void**)&weight_gpu, sizeof(float_t) * params.in_dim * params.out_dim));
      checkCudaErrors(cudaMalloc((void**)&dW_gpu, sizeof(float_t) * params.in_dim * params.out_dim));
      checkCudaErrors(cudaMalloc((void**)&prev_delta_gpu, sizeof(float_t) * params.batch_size * params.in_dim));
      checkCudaErrors(cudaMalloc((void**)&curr_delta_gpu, sizeof(float_t) * params.batch_size * params.out_dim));

      if (params.has_bias) {
        checkCudaErrors(cudaMalloc((void**)&db_gpu, sizeof(float_t) * params.out_dim));
        checkCudaErrors(cudaMalloc((void**)&onevec, sizeof(float_t) * params.batch_size));

        tensor_t ones({params.batch_size, 1, 1, 1});
        ones.fill(1.0f);
        cuda_push_array(onevec, &(*ones.host_begin()), ones.size());
      }
#endif
    }

    ~ConCudaBackwardGradOp() {
#ifdef USE_CUDNN
      cuda_free(prev_in_gpu);
      cuda_free(weight_gpu);
      cuda_free(dW_gpu);
      cuda_free(prev_delta_gpu);
      cuda_free(curr_delta_gpu);

      if (db_gpu) {
        cuda_free(db_gpu);
        cuda_free(onevec);
      }
#endif
    }

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
      cuda_push_array(prev_in_gpu, &(*prev_in.host_begin()), prev_in.size());
      cuda_push_array(weight_gpu, &(*weight.host_begin()), weight.size());
      cuda_push_array(curr_delta_gpu, &(*curr_delta.host_begin()), curr_delta.size());

      /** Backward propagate */
      checkCudaErrors(cublasSgemm(cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_T, params.in_dim, params.out_dim,
                                  params.batch_size, &alpha, prev_in_gpu, params.in_dim, curr_delta_gpu, params.out_dim,
                                  &beta, dW_gpu, params.in_dim));

      // scale due to batch size
      // float_t alpha = float_t(1) / static_cast<float_t>(params.batch_size);
      // checkCudaErrors(cublasSscal(cublas_handle(), dW.size(), &alpha, dW_gpu, one));

      // Data
      checkCudaErrors(cublasSgemm(cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N, params.in_dim, params.batch_size,
                                  params.out_dim, &alpha, weight_gpu, params.in_dim, curr_delta_gpu, params.out_dim,
                                  &beta, prev_delta_gpu, params.in_dim));

      if (params.has_bias) {
        checkCudaErrors(cublasSgemv(cublas_handle(), CUBLAS_OP_N, params.out_dim, params.batch_size, &alpha,
                                    curr_delta_gpu, params.out_dim, onevec, 1, &beta, db_gpu, 1));
        // checkCudaErrors(cublasSscal(cublas_handle(), db.size(), &alpha, db_gpu, one));
        checkCudaErrors(cudaDeviceSynchronize());
        cuda_pull_array(db_gpu, &(*db.host_begin()), db.size());
      }

      /** Pull result from device */
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
    float_t beta  = 0.0f;

    float_t* onevec = nullptr;
#endif
  };
}  // namespace simpleCNN

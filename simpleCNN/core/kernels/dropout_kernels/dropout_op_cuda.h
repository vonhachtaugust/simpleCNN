//
// Created by hacht on 5/22/17.
//

#pragma once

#include "../../framework/op_kernel.h"
#include "../../params/dropout_params.h"

#ifdef USE_CUDNN
#include "../../../../third_party/cudnn/include/cudnn.h"
#include "../../../util/cuda_utils.h"
#endif

namespace simpleCNN {

  class DropoutCudaForwardOp : public core::OpKernel {
   public:
    explicit DropoutCudaForwardOp(const core::OpKernelConstruction& context) : core::OpKernel(context) {
#ifdef USE_CUDNN
      core::Dropout_params* drop_params_ptr = static_cast<core::Dropout_params*>(core::OpKernel::params_);
      const auto& params                    = drop_params_ptr->dropout_params();

      if (product(params.shape) == 0) {
        throw simple_error("Drop layer shape not initialized");
      }

      checkCUDNN(cudnnCreateTensorDescriptor(&srcTensorDesc));
      checkCUDNN(cudnnSetTensor4dDescriptor(srcTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, params.shape[0],
                                            params.shape[1], params.shape[2], params.shape[3]));

      checkCUDNN(cudnnCreateTensorDescriptor(&dstTensorDesc));
      checkCUDNN(cudnnSetTensor4dDescriptor(dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, params.shape[0],
                                            params.shape[1], params.shape[2], params.shape[3]));

      checkCudaErrors(cudaMalloc((void**)&input_gpu, sizeof(float_t) * product(params.shape)));
      checkCudaErrors(cudaMalloc((void**)&output_gpu, sizeof(float_t) * product(params.shape)));

      reserveSpaceInBytes = get_reserve_space_size();
      stateSizeInBytes    = get_state_size();

      if (stateSizeInBytes > 0) {
        checkCudaErrors(cudaMalloc(&states, stateSizeInBytes));
      }

      if (reserveSpaceInBytes > 0) {
        checkCudaErrors(cudaMalloc(&reserveSpace, reserveSpaceInBytes));
      }

      checkCUDNN(cudnnCreateDropoutDescriptor(&dropDesc));
      checkCUDNN(
        cudnnSetDropoutDescriptor(dropDesc, cudnn_handle(), params.prob, states, stateSizeInBytes, time(NULL)));
#endif
    }

    ~DropoutCudaForwardOp() {
#ifdef USE_CUDNN
      checkCUDNN(cudnnDestroyTensorDescriptor(srcTensorDesc));
      checkCUDNN(cudnnDestroyTensorDescriptor(dstTensorDesc));
      checkCUDNN(cudnnDestroyDropoutDescriptor(dropDesc));

      cuda_free(input_gpu);
      cuda_free(output_gpu);

      if (reserveSpace) {
        checkCudaErrors(cudaFree(reserveSpace));
        reserveSpaceInBytes = 0;
      }

      if (states) {
        checkCudaErrors(cudaFree(states));
        stateSizeInBytes = 0;
      }
#endif
    }

    void compute(const core::OpKernelContext& context) override {
#ifdef USE_CUDNN
      const tensor_t& in_data = context.input(0);
      tensor_t& out_data      = context.output(0);

      /** Push to device memory */
      cuda_push_array(input_gpu, &(*in_data.host_begin()), in_data.size());

      /** Forward propagate */
      checkCUDNN(cudnnDropoutForward(cudnn_handle(), dropDesc, srcTensorDesc, input_gpu, dstTensorDesc, output_gpu,
                                     reserveSpace, reserveSpaceInBytes));

      /** Pull from device memory */
      checkCudaErrors(cudaDeviceSynchronize());
      cuda_pull_array(output_gpu, &(*out_data.host_begin()), out_data.size());
#endif
    }

   private:
#ifdef USE_CUDNN
    float_t* input_gpu  = nullptr;
    float_t* output_gpu = nullptr;

    cudnnTensorDescriptor_t srcTensorDesc;
    cudnnTensorDescriptor_t dstTensorDesc;
    cudnnDropoutDescriptor_t dropDesc;

    size_t reserveSpaceInBytes = 0;
    void* reserveSpace         = nullptr;

    size_t stateSizeInBytes = 0;
    void* states            = nullptr;

    size_t get_state_size() {
      size_t s = 0;
      checkCUDNN(cudnnDropoutGetStatesSize(cudnn_handle(), &s));
      return s;
    }

    size_t get_reserve_space_size() {
      size_t s = 0;
      checkCUDNN(cudnnDropoutGetReserveSpaceSize(srcTensorDesc, &s));
      return s;
    }
#endif
  };

  class DropoutCudaBackwardOp : public core::OpKernel {
   public:
    explicit DropoutCudaBackwardOp(const core::OpKernelConstruction& context) : core::OpKernel(context) {
#ifdef USE_CUDNN
      core::Dropout_params* drop_params_ptr = static_cast<core::Dropout_params*>(core::OpKernel::params_);
      const auto& params                    = drop_params_ptr->dropout_params();

      if (params.shape.size() == 0) {
        throw simple_error("Dropout layer shape not initialized");
      }

      checkCUDNN(cudnnCreateTensorDescriptor(&dsrcTensorDesc));
      checkCUDNN(cudnnSetTensor4dDescriptor(dsrcTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, params.shape[0],
                                            params.shape[1], params.shape[2], params.shape[3]));

      checkCUDNN(cudnnCreateTensorDescriptor(&ddstTensorDesc));
      checkCUDNN(cudnnSetTensor4dDescriptor(ddstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, params.shape[0],
                                            params.shape[1], params.shape[2], params.shape[3]));

      checkCudaErrors(cudaMalloc((void**)&curr_delta_gpu, sizeof(float_t) * product(params.shape)));
      checkCudaErrors(cudaMalloc((void**)&prev_delta_gpu, sizeof(float_t) * product(params.shape)));

      reserveSpaceInBytes = get_reserve_space_size();
      stateSizeInBytes    = get_state_size();

      if (stateSizeInBytes > 0) {
        checkCudaErrors(cudaMalloc(&states, stateSizeInBytes));
      }

      if (reserveSpaceInBytes > 0) {
        checkCudaErrors(cudaMalloc(&reserveSpace, reserveSpaceInBytes));
      }

      checkCUDNN(cudnnCreateDropoutDescriptor(&dropDesc));
      checkCUDNN(
        cudnnSetDropoutDescriptor(dropDesc, cudnn_handle(), params.prob, states, stateSizeInBytes, time(NULL)));
#endif
    }

    ~DropoutCudaBackwardOp() {
#ifdef USE_CUDNN
      checkCUDNN(cudnnDestroyTensorDescriptor(dsrcTensorDesc));
      checkCUDNN(cudnnDestroyTensorDescriptor(ddstTensorDesc));
      checkCUDNN(cudnnDestroyDropoutDescriptor(dropDesc));

      cuda_free(curr_delta_gpu);
      cuda_free(prev_delta_gpu);

      if (reserveSpace) {
        checkCudaErrors(cudaFree(reserveSpace));
        reserveSpaceInBytes = 0;
      }

      if (states) {
        checkCudaErrors(cudaFree(states));
        stateSizeInBytes = 0;
      }
#endif
    }

    void compute(const core::OpKernelContext& context) override {
#ifdef USE_CUDNN
      const tensor_t& curr_delta = context.output_grad(0);
      tensor_t& prev_delta       = context.input_grad(0);

      /** Push to device memory */
      cuda_push_array(curr_delta_gpu, &(*curr_delta.host_begin()), curr_delta.size());

      /** Forward propagate */
      checkCUDNN(cudnnDropoutForward(cudnn_handle(), dropDesc, ddstTensorDesc, curr_delta_gpu, dsrcTensorDesc,
                                     prev_delta_gpu, reserveSpace, reserveSpaceInBytes));

      /** Pull from device memory */
      checkCudaErrors(cudaDeviceSynchronize());
      cuda_pull_array(prev_delta_gpu, &(*prev_delta.host_begin()), prev_delta.size());
#endif
    }

   private:
#ifdef USE_CUDNN
    float_t* curr_delta_gpu;
    float_t* prev_delta_gpu;

    cudnnTensorDescriptor_t dsrcTensorDesc;
    cudnnTensorDescriptor_t ddstTensorDesc;
    cudnnDropoutDescriptor_t dropDesc;

    size_t reserveSpaceInBytes = 0;
    void* reserveSpace         = nullptr;

    size_t stateSizeInBytes = 0;
    void* states            = nullptr;

    size_t get_state_size() {
      size_t s = 0;
      checkCUDNN(cudnnDropoutGetStatesSize(cudnn_handle(), &s));
      return s;
    }

    size_t get_reserve_space_size() {
      size_t s = 0;
      checkCUDNN(cudnnDropoutGetReserveSpaceSize(dsrcTensorDesc, &s));
      return s;
    }
#endif
  };
}  // namespace simpleCNN
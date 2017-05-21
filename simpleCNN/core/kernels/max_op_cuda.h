//
// Created by hacht on 5/21/17.
//

#pragma once

#include "../params/max_params.h"
#include "max_op_internal.h"

namespace simpleCNN {
  class MaxpoolingCudaForwardOp : public core::OpKernel {
   public:
    explicit MaxpoolingCudaForwardOp(const core::OpKernelConstruction& context) : core::OpKernel(context) {}

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
      checkCUDNN(cudnnPoolingForward(params.cudnnHandle, params.poolDesc, &one, params.srcTensorDesc, in_data_gpu, &one,
                                     params.dstTensorDesc, out_data_gpu));

      /** Pull result from device */
      cuda_pull_array(out_data_gpu, &(*out_data.host_begin()), out_data.size());

      /** Release allocated gpu memory */
      cuda_free(in_data_gpu);
      cuda_free(out_data_gpu);
#endif
    }
  };

  class MaxpoolingCudaBackwardOp : public core::OpKernel {
   public:
    explicit MaxpoolingCudaBackwardOp(const core::OpKernelConstruction& context) : core::OpKernel(context) {}

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
      checkCUDNN(cudnnPoolingBackward(params.cudnnHandle, params.poolDesc, &one, params.dstTensorDesc, out_data_gpu,
                                      params.ddstTensorDesc, curr_delta_gpu, params.srcTensorDesc, in_data_gpu, &one,
                                      params.dsrcTensorDesc, prev_delta_gpu));

      /** Pull result from device */
      cuda_pull_array(prev_delta_gpu, &(*prev_delta.host_begin()), prev_delta.size());

      /** Release allocated gpu memory */
      cuda_free(in_data_gpu);
      cuda_free(out_data_gpu);
      cuda_free(curr_delta_gpu);
      cuda_free(prev_delta_gpu);
#endif
    }
  };
}  // namespace simpleCNN
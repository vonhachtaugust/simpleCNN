//
// Created by hacht on 3/4/17.
//

#pragma once

#include "../../layers/layer.h"
#include "../../util/util.h"
#include "../backend.h"
#include "../params/params.h"
#include "device.h"
#include "tensor.h"

namespace simpleCNN {
  namespace core {

    class OpKernel;

    class OpKernelConstruction {
     public:
      explicit OpKernelConstruction(Device* device, Params* params) : device_(device), params_(params) {}

      // Returns the device raw pointer
      Device* device() const { return device_; }

      // Returns the device raw pointer
      Params* params() const { return params_; }

     private:
      Device* device_ = nullptr;
      Params* params_ = nullptr;
    };

    class OpKernelContext {
     public:
      struct OpParams {
        // the operation kernel being computed
        OpKernel* op_kernel_ptr_ = nullptr;

        // the device on which the kernel is running
        Device* device_ptr_ = nullptr;

        // the layer on which the kernel is running
        Layer* layer_ptr_ = nullptr;

        // the operation parameters
        Params* params_ptr_ = nullptr;

        backend_t engine = default_engine();
      };

      explicit OpKernelContext(const data_ptrs_t& in_data, data_ptrs_t& out_data)
        : in_data_(in_data), out_data_(out_data) {
        op_params_ = std::unique_ptr<OpParams>(new OpParams());
      }

      explicit OpKernelContext(const data_ptrs_t& in_data,
                               const data_ptrs_t& out_data,
                               data_ptrs_t& in_grad,
                               data_ptrs_t& out_grad)
        : in_data_(in_data), out_data_(out_data), out_grad_(out_grad), in_grad_(in_grad) {
        op_params_ = std::unique_ptr<OpParams>(new OpParams());
      }

      // Getters and setters
      tensor_t& input(const int idx) const { return *in_data_[idx]; }

      tensor_t& output(const int idx) const { return *out_data_[idx]; }

      tensor_t& input_grad(const int idx) const { return *in_grad_[idx]; }

      tensor_t& output_grad(const int idx) const { return *out_grad_[idx]; }

      void setParams(Params* params) { op_params_->params_ptr_ = std::move(params); }

      // Backend
      void setEngine(const backend_t engine) { op_params_->engine = engine; }

      backend_t engine() const { return op_params_->engine; }

     private:
      data_ptrs_t in_data_;
      data_ptrs_t out_data_;
      data_ptrs_t out_grad_;
      data_ptrs_t in_grad_;

      std::unique_ptr<OpParams> op_params_;
    };

    class OpKernel {
     public:
      explicit OpKernel(const OpKernelConstruction& context) : device_(context.device()), params_(context.params()) {}

      virtual ~OpKernel() {}

      /**
       * Computes the resulting affine transformation of a forward/backward pass.
       *
       * @param context : Object holding input data, output data, layer type etc.
       */
      virtual void compute(const OpKernelContext& context) = 0;

     protected:
      Device* device_ = nullptr;
      Params* params_ = nullptr;
    };
  }  // namespace core
}  // namespace simpleCNN

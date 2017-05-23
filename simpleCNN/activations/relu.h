//
// Created by hacht on 5/8/17.
//

#pragma once

namespace simpleCNN {
  namespace activation {
    class ReLU : public Activation_layer {
     public:
      ReLU(const core::backend_t backend_type = core::default_engine()) : Activation_layer(backend_type) {}
      ReLU(const shape4d& shape, const core::backend_t backend_type = core::default_engine())
        : Activation_layer(shape, backend_type) {}

      std::string layer_type() const override { return "ReLU-activation-layer"; }

      template <typename T>
      T f(const T& value) const {
        return std::max(T{0}, value);
      }

      template <typename T>
      T df(T value) const {
        return value > T{0} ? T{1} : T{0};
      }

      void forward_activation(const tensor_t& affine, tensor_t& activated) const override {
        for (size_t i = 0; i < affine.size(); ++i) {
          activated.host_at_index(i) = f(affine.host_at_index(i));
        }
      }

      void backward_activation(const tensor_t& affine, const tensor_t& curr_delta, tensor_t& activated) const override {
        for (size_t i = 0; i < affine.size(); ++i) {
          activated.host_at_index(i) = df(affine.host_at_index(i)) * curr_delta.host_at_index(i);
        }
      };

      void forward_activation_gpu(const tensor_t& affine, tensor_t& activated) const override {
#ifdef USE_CUDNN
        /** Initialize device memory */
        float_t* affine_gpu    = cuda_make_array(&(*affine.host_begin()), affine.size());
        float_t* activated_gpu = cuda_make_array(&(*activated.host_begin()), activated.size());

        /** Forward propagate */
        float_t one = 1;
        checkCUDNN(cudnnActivationForward(Activation_layer::cudnnHandle, Activation_layer::Activation, &one,
                                          Activation_layer::srcTensorDesc, affine_gpu, &one,
                                          Activation_layer::dstTensorDesc, activated_gpu));

        /** Pull result from device */
        checkCudaErrors(cudaDeviceSynchronize());
        cuda_pull_array(activated_gpu, &(*activated.host_begin()), activated.size());

        /** Release allocated gpu memory */
        cuda_free(affine_gpu);
        cuda_free(activated_gpu);
#endif
      }

      void backward_activation_gpu(const tensor_t& affine,
                                   const tensor_t& activated,
                                   const tensor_t& curr_delta,
                                   tensor_t& prev_delta) const override {
#ifdef USE_CUDNN
        /** Initialize device memory */
        float_t* affine_gpu     = cuda_make_array(&(*affine.host_begin()), affine.size());
        float_t* activated_gpu  = cuda_make_array(&(*activated.host_begin()), activated.size());
        float_t* curr_delta_gpu = cuda_make_array(&(*curr_delta.host_begin()), curr_delta.size());
        float_t* prev_delta_gpu = cuda_make_array(&(*prev_delta.host_begin()), prev_delta.size());

        /** Backward propagate */
        float_t one = 1;
        checkCUDNN(cudnnActivationBackward(
          Activation_layer::cudnnHandle, Activation_layer::Activation, &one, Activation_layer::dstTensorDesc,
          activated_gpu, Activation_layer::ddstTensorDesc, curr_delta_gpu, Activation_layer::srcTensorDesc, affine_gpu,
          &one, Activation_layer::dsrcTensorDesc, prev_delta_gpu));

        /**  Pull result from device */
        checkCudaErrors(cudaDeviceSynchronize());
        cuda_pull_array(prev_delta_gpu, &(*prev_delta.host_begin()), prev_delta.size());

        /** Release allocated gpu memory */
        cuda_free(affine_gpu);
        cuda_free(activated_gpu);
        cuda_free(curr_delta_gpu);
        cuda_free(prev_delta_gpu);
#endif
      };

      std::pair<float_t, float_t> scale() const override { return std::make_pair(float_t(0), float_t(1)); };
    };
  }  // namespace activation
}  // namespace simpleCNN

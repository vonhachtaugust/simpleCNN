//
// Created by hacht on 4/26/17.
//

#pragma once

#include "../network.h"
namespace simpleCNN {

  class Dropout_layer : public Layer {
   public:
    Dropout_layer(shape4d in_dim, float_t prob, net_phase phase = net_phase::train)
      : Layer({tensor_t(component_t::IN_DATA)}, {tensor_t(component_t::OUT_DATA)}),
        shape_(in_dim),
        in_size_(product(in_dim)),
        phase_(phase),
        prob_(prob),
        mask_(in_dim) {}

    void set_prob(float_t prob) { prob_ = prob; }

    void forward_activation(const tensor_t& affine, tensor_t& activated) { return; }

    void backward_activation(const tensor_t& prev_delta, const tensor_t& affine, tensor_t& activated) { return; }

    shape_t in_shape() const override { return {shape_}; }

    shape_t out_shape() const override { return {shape_}; }

    void forward_propagation(const data_ptrs_t& in_data, data_ptrs_t& out_data) override {
      const tensor_t& in = *in_data[0];
      tensor_t& out      = *out_data[0];

      const size_t n = in.size();

      if (phase_ == net_phase::train) {
        for (size_t i = 0; i < n; ++i) {
          mask_.host_at_index(i) = bernoulli(prob_);
          out.host_at_index(i)   = in.host_at_index(i) * mask_.host_at_index(i);
        }
      } else {
        for (size_t i = 0; i < n; ++i) {
          // Approximate the output by expected input value.
          out.host_at_index(i) = in.host_at_index(i) * prob_;
        }
      }
    }

    void back_propagation(const data_ptrs_t& in_data,
                          const data_ptrs_t& out_data,
                          data_ptrs_t& in_grad,
                          data_ptrs_t& out_grad) override {
      const tensor_t& curr_grad = *out_grad[0];
      tensor_t& prev_grad = *in_grad[0];

      const size_t n = curr_grad.size();

      for (size_t i = 0; i < n; ++i) {
        prev_grad.host_at_index(i) = mask_.host_at_index(i) * curr_grad.host_at_index(i);
      }
    }

    void set_net_phase(net_phase phase) { phase_ = phase; }

    std::string layer_type() const override { return "dropout"; }

   private:
    shape4d shape_;
    size_t in_size_;
    net_phase phase_;
    float_t prob_;
    tensor_t mask_;
  };
}  // namespace simpleCNN

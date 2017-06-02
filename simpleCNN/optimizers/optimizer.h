//
// Created by hacht on 3/27/17.
//

#pragma once

#include <unordered_map>
#include "../core/framework/tensor_utils.h"
#include "../util/util.h"

namespace simpleCNN {
  class Optimizer {
   public:
    virtual ~Optimizer() = default;

    virtual void reset() {}
    /**
     * Method to update a parameter p.
     *
     * @param dp - parameter gradient values.
     * @param p - current parameter values, to be updated.
     */
    // virtual void update_weight(const tensor_t* dW, tensor_t* W) = 0;

    // virtual void update_bias(const tensor_t* dB, tensor_t* B) = 0;

    virtual void update(const tensor_t* dW, const tensor_t* dB, tensor_t* W, tensor_t* B, const size_t batch_size) = 0;
  };

  /**
   * Helper class to keep track of weight and bias gradient history.
   *
   * @tparam N : number of lists of values to remember history of.
   */
  template <size_t N>
  class Stateful_optimizer : public Optimizer {
   public:
    void reset() override {
      for (auto& e : state) {
        e.clear();
      }
    }

   protected:
    template <size_t index>
    tensor_t* get(const tensor_t* key) {
      static_assert(index < N, "Index out of range");

      if (!state[index][key]) {
        keys.push_back(std::make_shared<tensor_t>(key->shape_v()));
        state[index][key] = keys.back().get();
      }
      return state[index][key];
    }
    std::unordered_map<const tensor_t*, tensor_t*> state[N];
    std::vector<std::shared_ptr<tensor_t>> keys;
  };

  template <typename T = float_t>
  class Adam : public Stateful_optimizer<4> {
   public:
    Adam()
      : alpha(Hyperparameters::learning_rate),
        eps(float_t(1e-8)),
        beta1(float_t(0.9)),
        beta2(float_t(0.999)),
        beta1_t(float_t(0.9)),
        beta2_t(float_t(0.999)) {}

    void update(const tensor_t* dW, const tensor_t* dB, tensor_t* W, tensor_t* B, const size_t batch_size) {
      tensor_t* mt_w = get<0>(W);
      tensor_t* vt_w = get<1>(W);
      tensor_t* mt_b = get<0>(B);
      tensor_t* vt_b = get<1>(B);

      beta1_t *= beta1;
      beta2_t *= beta2;

      adam(dW, mt_w, vt_w, W, batch_size, true);
      adam(dB, mt_b, vt_b, B, batch_size, false);
    }

    void adam(const tensor_t* dx,
              const tensor_t* mt,
              const tensor_t* vt,
              tensor_t* x,
              const size_t batch_size,
              const bool weight_decay) {
      for (size_t i = 0; i < dx->size(); ++i) {
        auto& mt_i = mt->host_at_index(i);
        auto& vt_i = vt->host_at_index(i);
        auto& dx_i = dx->host_at_index(i);
        auto& x_i  = x->host_at_index(i);

        dx_i /= float_t(batch_size);

        mt_i = beta1 * mt_i + (float_t(1) - beta1) * dx_i;
        vt_i = beta2 * vt_i + (float_t(1) - beta2) * dx_i * dx_i;

        float_t adaptive_lr = alpha * std::sqrt(1 - beta2_t) / (float_t(1) - beta1_t);

        if (weight_decay) {
          x_i -= adaptive_lr * mt_i / (std::sqrt(vt_i) + eps) + Hyperparameters::regularization_constant * x_i;
        } else {
          x_i -= adaptive_lr * mt_i / (std::sqrt(vt_i) + eps);
        }
      }
    }

   private:
    float_t beta1;    // decay term
    float_t beta2;    // decay term
    float_t beta1_t;  // decay term over time
    float_t beta2_t;  // decay term over time
    float_t alpha;    // learning rate
    float_t eps;      // dummy term to avoid division by zero
  };
}
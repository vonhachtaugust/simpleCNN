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

    /**
     * Method to update a parameter p.
     *
     * @param dp - parameter gradient values.
     * @param p - current parameter values, to be updated.
     */
    virtual void update(const tensor_t* dp, tensor_t* p) = 0;
  };

  /**
   * Helper class to keep track of weight gradient history.
   *
   * @tparam N : number of lists of values to remember history of.
   */
  template <size_t N>
  class Stateful_optimizer : public Optimizer {
   protected:
    template <size_t index>
    tensor_t* get(const tensor_t* key) {
      static_assert(index < N, "Index out of range");

      if (!state[index][key]) {
        std::vector<size_t> shape(key->shape().size());
        std::copy(key->shape().begin(), key->shape().end(), shape.begin());
        keys.push_back(std::make_shared<tensor_t>(shape));
        state[index][key] = keys.back().get();
      }

      return state[index][key];
    }
    std::unordered_map<const tensor_t*, tensor_t*> state[N];
    std::vector<std::shared_ptr<tensor_t>> keys;
  };

  template <typename T = float_t>
  class Adam : public Stateful_optimizer<2> {
   public:
    Adam()
      : alpha(float_t(0.001)),
        b1(float_t(0.9)),
        b2(float_t(0.999)),
        b1_t(float_t(0.9)),
        b2_t(float_t(0.999)),
        eps(float_t(1e-8)) {}

    void update(const tensor_t* dp, tensor_t* p) override {
      assert(dp->size() == p->size());
      tensor_t* mt = get<0>(p);
      tensor_t* vt = get<1>(p);

      b1_t *= b1;
      b2_t *= b2;

      for (size_t i = 0; i < dp->size(); ++i) {
        auto& mt_i = mt->host_index(i);
        auto& vt_i = vt->host_index(i);
        auto& dp_i = dp->host_index(i);
        auto& p_i  = p->host_index(i);

        mt_i = b1 * mt_i + (float_t(1) - b1) * dp_i;
        vt_i = b2 * vt_i + (float_t(1) - b2) * dp_i * dp_i;

        p_i -= alpha * (mt_i / (float_t(1) - b1_t)) / std::sqrt((vt_i / (float_t(1) - b2_t)) + eps);
      }
    }

   private:
    float_t alpha;  // learning rate
    float_t b1;     // decay term
    float_t b2;     // decay term
    float_t b1_t;   // decay term power t
    float_t b2_t;   // decay term power t
    T eps;
  };
}
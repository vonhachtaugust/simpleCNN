//
// Created by hacht on 5/22/17.
//

#pragma once

#include "../../util/util.h"
#include "../framework/op_kernel.h"
#include "params.h"

namespace simpleCNN {
  namespace core {

    enum class activation_t { relu, tanh, softmax };

    class Activation_params : public Params {
     public:
      shape4d shape;
      core::activation_t activation_function;

      const Activation_params& activation_params() const { return *this; }
    };
  }
}
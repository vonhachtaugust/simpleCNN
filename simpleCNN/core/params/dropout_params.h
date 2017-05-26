//
// Created by hacht on 5/22/17.
//

#pragma once

#include "../../util/util.h"
#include "params.h"

namespace simpleCNN {
  namespace core {

    class Dropout_params : public Params {
     public:
      shape4d shape;
      size_t in_size;
      net_phase phase;
      float_t prob;
      tensor_t mask;

      const Dropout_params &dropout_params() const { return *this; }
    };

  }  // namespace core
}  // namespace simpleCNNM
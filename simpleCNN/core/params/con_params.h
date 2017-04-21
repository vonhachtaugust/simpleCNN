//
// Created by hacht on 4/5/17.
//

#pragma once

#include "params.h"

namespace simpleCNN {
  namespace core {

    class Con_params : public Params {
     public:
      // Input parameters
      size_t in_dim;
      size_t out_dim;
      size_t batch_size;
      bool has_bias;

      const Con_params& connected_params() const { return *this; }
    };
  }
}
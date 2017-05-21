//
// Created by hacht on 3/4/17.
//

#pragma once

#include <iostream>
#include "../util/simple_error.h"

namespace simpleCNN {
  namespace core {
    enum class backend_t { internal, gpu };

    inline backend_t default_engine() { return backend_t::internal; }

    class Backend {
     public:
      Backend() {}

      // core math functions --------------------------------- //

      // ----------------------------------------------------- //

      virtual backend_t type() const = 0;

     protected:
    };
  }  // namespace core
}  // namespace simpleCNN

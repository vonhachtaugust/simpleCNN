//
// Created by hacht on 3/4/17.
//

#pragma once

namespace simpleCNN {
    namespace core {
        class Conv_params;

        class Params {
        public:
            Params() { }

            virtual Conv_params conv();
        };
    }
} // namespace simpleCNN

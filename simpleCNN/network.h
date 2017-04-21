//
// Created by hacht on 4/18/17.
//

#pragma once

#include "lossfunctions/loss_functions.h"
#include "network_types.h"

namespace simpleCNN {

  typedef int label_t;

  template <typename NetType>
  class Network {
   public:
    explicit Network() : stop_training_(false) {}

    template <typename loss, typename optimizer>
    bool train(Optimizer& opt,
               const tensor_t& input,   // images corresponding to these targets
               const tensor_t& target,  // selected training data targets
               const std::vector<label_t>& class_lables,
               size_t batch_size,
               size_t epoch,
               const bool reset_weight) {
      if (input.size() != class_lables.size()) {
        return false;
      }
      if (input.size() < batch_size || class_lables.size() < batch_size) {
        return false;
      }

      tensor_t output_delta({batch_size, 1, target.dimension(dim_t::height), 0});

      train_onebatch<loss, optimizer>(opt, input, target, output_delta, batch_size);

      return true;
    }

   private:
    /**
     * trains on one minibatch, i.e. runs forward and backward propagation to
     * calculate
     * the gradient of the loss function with respect to the network parameters
     * (weights),
     * then calls the optimizer algorithm to update the weights
     *
     * @param batch_size the number of data points to use in this batch
     */
    template <typename loss, typename optimizer>
    void train_onebatch(
      optimizer& opt, const tensor_t& in, const tensor_t& target, tensor_t& output_delta, const size_t batch_size) {
      auto output = forward_pass(in);
      backward_pass<loss>(output, target, output_delta, batch_size);
      net_.update_weights(opt);
    }

    tensor_t forward_pass(const tensor_t& in) { net_.forward(in); }

    template <typename loss>
    void backward_pass(const tensor_t& output,
                       const tensor_t& target,
                       tensor_t& output_delta,
                       const size_t batch_size) {
      gradient<loss>(output, target, output_delta, batch_size);
      net_.backward(output_delta);
    }

    template <typename layer>
    friend Network<Sequential>& operator<<(Network<Sequential>& n, layer&& l);

    NetType net_;
    bool stop_training_;
  };

  template <typename layer>
  Network<Sequential>& operator<<(Network<Sequential>& n, layer&& l) {
    n.net_.add(std::forward<layer>(l));
    return n;
  }

}  // namespace simpleCNN

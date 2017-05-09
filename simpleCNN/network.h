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

    template<typename loss>
    void gradient_check(const tensor_t& input, const tensor_t& labels, const size_t batch_size) {
      net_.setup(true);
      tensor_t output = net_.forward(input);
      print(output, "Output");
      tensor_t delta = gradient<loss>(output, labels, batch_size);
      net_.backward(delta);
      //net_.print_layers();
    }

    /**
     * Test a forward pass and check the result.
     * Used to check if weight init was appropriate (i.e. no saturation)
     *
     * @param input : test batch
     * @return network output
     */
    tensor_t test(const tensor_t& input) {
      net_.setup(true);
      return net_.forward(input);
    };

    /**
     * Test a forward pass and check the result
     * Used to check if weight init was appropriate (i.e. loss is not infinity)
     *
     * @tparam Loss : Loss function
     * @param input : test batch
     * @param target : list of correct labels
     * @param batch_size
     */
    template <typename Loss>
    void test_loss(const tensor_t& input, const tensor_t& target, const size_t batch_size) {
      net_.setup(true);
      tensor_t output = net_.forward(input);
      float_t er      = error<Loss>(output, target, batch_size);
      print(er, "Error: ");
    }

    /**
     * Test forward pass twice.
     * Used to check if forward pass returns the same output given the same input.
     *
     * @tparam Loss
     * @param input
     * @param target
     * @param batch_size
     */
    template <typename Loss>
    void test_forward_twice(const tensor_t& input) {
      net_.setup(true);
      tensor_t output = net_.forward(input);
      print(output, "Output_I");
      output = net_.forward(input);
      print(output, "Output_II");
    }

    /**
     * Test a forward pass, backward pass and finally a forward pass.
     * Used to check if loss seems to reduce.
     *
     * @tparam Loss : Loss function
     * @tparam optimizer : optimizer function
     * @param opt : optimizer instance
     * @param in : test batch image
     * @param target : list of correct labels
     * @param output_delta : store delta results to check gradient values
     * @param batch_size
     */
    template <typename Loss, typename optimizer>
    void test_onbatch(
      optimizer& opt, const tensor_t& in, const tensor_t& target, tensor_t& output_delta, const size_t batch_size) {
      net_.setup(true);
      tensor_t output = net_.forward(in);
      print(error<Loss>(output, target, batch_size), "Error: ");
      gradient<Loss>(output, target, output_delta, batch_size);
      net_.backward(output_delta);
      net_.update(opt, batch_size);
      output = net_.forward(in);
      print(error<Loss>(output, target, batch_size), "Error: ");
    };

    template <typename loss, typename optimizer, typename OnBatchEnumerate, typename OnEpochEnumerate>
    bool train(optimizer& opt,
               const tensor_t& input,
               const tensor_t& train_labels,
               size_t batch_size,
               size_t epoch,
               OnBatchEnumerate on_batch_enumerate,
               OnEpochEnumerate on_epoch_enumerate,
               const bool reset_weight = false) {
      if (!(input.size() >= train_labels.size())) {
        return false;
      }
      if (input.size() < batch_size || train_labels.size() < batch_size) {
        return false;
      }
      net_.setup(reset_weight);
      set_netphase(net_phase::train);
      opt.reset();
      stop_training_ = false;
      time_t t = clock();

      for (size_t i = 0; i < epoch && !stop_training_; ++i) {
        for (size_t j = 0; j < input.size() && !stop_training_; j += batch_size) {
          train_once<loss>(opt, input.subView({j}, {batch_size, input.dimension(dim_t::depth),
                                                    input.dimension(dim_t::height), input.dimension(dim_t::width)}),
                           train_labels.subView({j}, {batch_size, 1, 1, 1}), batch_size);
          on_batch_enumerate(t);
        }
        on_epoch_enumerate(i);
      }
      return true;
    }

    void set_netphase(net_phase phase) {
      for (auto n : net_) {
        n->set_netphase(phase);
      }
    }

   private:
    template <typename loss, typename optimizer>
    void train_once(optimizer& opt, const tensor_t minibatch, const tensor_t labels, const size_t batch_size) {
      tensor_t output = net_.forward(minibatch);
      print_seq(error<loss>(output, labels, batch_size));
      tensor_t grad = gradient<loss>(output, labels, batch_size);
      net_.backward(grad);
      net_.update(opt, batch_size);
    }

    template <typename loss, typename optimizer>
    void train_onebatch(optimizer& opt, const tensor_t* in, const tensor_t* target, const size_t batch_size) {
      tensor_t output = net_.forward(in);
      tensor_t grad   = gradient<loss>(output, *target, batch_size);
      net_.backward(grad);
      net_.update(opt, batch_size);
    };

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
      optimizer& opt, const tensor_t* in, const tensor_t* target, tensor_t* output_delta, const size_t batch_size) {
      tensor_t output = net_.forward(in);
      print(error<loss>(output, *target, batch_size), "Error: ");
      backward_pass<loss>(output, *target, *output_delta, batch_size);
      net_.update(opt, batch_size);
    }

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
    net_phase phase_;
    bool stop_training_;
  };

  template <typename layer>
  Network<Sequential>& operator<<(Network<Sequential>& n, layer&& l) {
    n.net_.add(std::forward<layer>(l));
    return n;
  }
}  // namespace simpleCNN

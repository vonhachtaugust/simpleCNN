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
    template<typename Loss>
    void test_loss(const tensor_t& input, const tensor_t& target, const size_t batch_size) {
      net_.setup(true);
      tensor_t output = net_.forward(input);
      float_t er = error<Loss>(output, target, batch_size);
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
    template<typename Loss>
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
    template<typename Loss, typename optimizer>
    void test_onbatch(optimizer& opt,
                      const tensor_t& in,
                      const tensor_t& target,
                      tensor_t& output_delta,
                      const size_t batch_size) {
      net_.setup(true);
      tensor_t output = net_.forward(in);
      print(error<Loss>(output, target, batch_size), "Error: ");
      gradient<Loss>(output, target, output_delta, batch_size);
      net_.backward(output_delta);
      net_.update(opt, batch_size);
      output = net_.forward(in);
      print(error<Loss>(output, target, batch_size), "Error: ");
    };

//    template<typename loss, typename optimizer, typename OnBatchEnumerate, typename OnEpochEnumaerate>
    template<typename loss, typename optimizer>
    bool test_mnist(optimizer& opt,
                    const tensor_t& train_data,
                    const std::vector<label_t>& train_lables,
                    size_t batch_size,
                    size_t mini_batch_size,
  //                  OnBatchEnumerate on_batch_enumerate,
  //                  OnEpochEnumaerate on_epoch_enumerate,
                    const bool reset_weight = true) {
      size_t in_w = train_data.shape()[0];
      size_t in_h = train_data.shape()[1];
      size_t num_classes = 10;

      tensor_t test_batch({batch_size, 1, in_h, in_w});
      tensor_t test_labels({batch_size, 1, 1, 1});
      tensor_t output_error({batch_size, 1, num_classes, 1});

      size_t n = batch_size * in_h * in_w;
      size_t m = batch_size;

      set_netphase(net_phase::train);
      net_.setup(reset_weight);
      opt.reset();
      stop_training_ = false;

      for (size_t i = 0; i < mini_batch_size; ++i) {
        for (size_t j = 0; j < n; ++j) {
          size_t index = n * i + j;
          test_batch.host_at_index(index) = train_data.host_at_index(index);
        }
        for (size_t j = 0; j < m; ++j) {
          size_t index = m * i + j;
          test_labels.host_at_index(index) = train_lables[index];
        }

        train_onebatch<loss, optimizer>(opt, test_batch, test_labels, output_error, batch_size);
      }
    };

    template <typename loss, typename optimizer, typename OnBatchEnumerate, typename OnEpochEnumerate>
    bool train(Optimizer& opt,
               const tensor_t& input,
               const tensor_t& target,
               const std::vector<label_t>& train_lables,
               size_t batch_size,
               size_t epoch,
               OnBatchEnumerate on_batch_enumerate,
               OnEpochEnumerate on_epoch_enumerate,
               const bool reset_weight = false) {
      if (!(input.size() >= train_lables.size())) {
        return false;
      }
      if (input.size() < batch_size || train_lables.size() < batch_size) {
        return false;
      }

      return true;
    }

    void set_netphase(net_phase phase) {
      for (auto n : net_) {
        n->set_netphase(phase);
      }
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
      print(error<loss>(output, target, batch_size), "Error: ");
      backward_pass<loss>(output, target, output_delta, batch_size);
      net_.update(opt, batch_size);
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
    net_phase phase_;
    bool stop_training_;
  };

  template <typename layer>
  Network<Sequential>& operator<<(Network<Sequential>& n, layer&& l) {
    n.net_.add(std::forward<layer>(l));
    return n;
  }
}  // namespace simpleCNN

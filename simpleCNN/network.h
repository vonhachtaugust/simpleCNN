//
// Created by hacht on 4/18/17.
//

#pragma once

#include <fstream>
#include "../third_party/cereal/archives/binary.hpp"
#include "../third_party/cereal/cereal.hpp"
#include "loss/loss_functions.h"
#include "network_types.h"

namespace simpleCNN {

  typedef int label_t;

  enum class content_type { weights_and_bias, loss, accuracy };

  enum class file_format { binary };

  template <typename NetType>
  class Network {
   public:
    explicit Network() : stop_training_(false) {}

    void print_weights_and_bias() {
      auto w = net_.get_weights();
      auto b = net_.get_bias();
      printvt_ptr(w, "Weights");
      printvt_ptr(b, "Bias");
    }

    void load_from_file(const std::string& filename,
                        content_type what,
                        file_format format = file_format::binary) const {
      std::ifstream ifs(filename.c_str(), std::ios::binary);

      if (ifs.fail() || ifs.bad()) {
        throw simple_error("Failed to open: " + filename);
      }

      from_archive(ifs, what);
    }

    void save_to_file(const std::string& filename, content_type what, file_format format = file_format::binary) const {
      std::ofstream ofs(filename.c_str(), std::ios::binary);

      if (ofs.fail() || ofs.bad()) {
        throw simple_error("Failed to open: " + filename);
      }

      to_archive(ofs, what);
    }

    template <typename OutputArchive>
    void to_archive(OutputArchive& ar, content_type what) const {
      if (what == content_type::weights_and_bias) {
        net_.save_weight_and_bias(ar);
      }
    }

    template <typename InputArchive>
    void from_archive(InputArchive& ar, content_type what) const {
      if (what == content_type::weights_and_bias) {
        net_.load_weight_and_bias(ar);
      }
    }

    std::vector<float_t> gradient_check(const tensor_t& input, const tensor_t& labels) {
      net_.setup(true);
      Adam<float_t> a;
      auto ng = computeNumericalGradient(input, labels);

      net_.forward_pass(input);
      net_.backward(labels);
      auto dW = net_.get_dW();

      //printvt(ng, "Numerical gradient");
      //printvt_ptr(dW, "Backprop gradient");

      return relative_error(dW, ng);
    }

    std::vector<float_t> gradient_check_bias(const tensor_t& input, const tensor_t& labels) {
      net_.setup(true);
      auto ng = computeNumericalGradient_bias(input, labels);

      net_.forward_pass(input);
      net_.backward(labels);
      auto dB = net_.get_dB();

      //printvt(ng, "Numerical gradient");
      //printvt_ptr(dB, "Backprop gradient");

      return relative_error(dB, ng);
    }

    /**
     * Perturb each weight by +epsilon / -epsilon and compute loss.
     * Use this to approximate gradient.
     */
    std::vector<tensor_t> computeNumericalGradient(const tensor_t& input, const tensor_t& labels) {
      std::vector<tensor_t*> weights = net_.get_weights();
      size_t batch_size              = input.shape()[0];
      std::vector<tensor_t> num_grads;

      float_t e = 1E-3;
      // For each layer containing weights.
      for (size_t i = 0; i < weights.size(); ++i) {
        // std::cout << "Computing ... " + std::to_string(i) << std::endl;
        // For each weight in the layer.
        tensor_t numerical_weight_gradient(weights[i]->shape_v());

        for (size_t j = 0; j < weights[i]->size(); ++j) {
          weights[i]->host_at_index(j) += e;
          auto loss1 = net_.forward_loss(input, labels);

          weights[i]->host_at_index(j) -= 2 * e;
          auto loss2 = net_.forward_loss(input, labels);

          numerical_weight_gradient.host_at_index(j) = (loss1 - loss2) / (2 * e);
          weights[i]->host_at_index(j) += e;
        }
        num_grads.push_back(numerical_weight_gradient);
      }
      return num_grads;
    }

    std::vector<tensor_t> computeNumericalGradient_bias(const tensor_t& input, const tensor_t& labels) {
      std::vector<tensor_t *> bias = net_.get_bias();
      size_t batch_size = input.shape()[0];
      std::vector<tensor_t> num_grads;

      float_t e = 1E-3;

      // For each layer containing weights.
      for (size_t i = 0; i < bias.size(); ++i) {
        // std::cout << "Computing ... " + std::to_string(i) << std::endl;
        // For each weight in the layer.
        tensor_t numerical_gradient(bias[i]->shape_v());

        for (size_t j = 0; j < bias[i]->size(); ++j) {
          bias[i]->host_at_index(j) += e;
          auto loss1 = net_.forward_loss(input, labels);

          bias[i]->host_at_index(j) -= 2 * e;
          auto loss2 = net_.forward_loss(input, labels);

          numerical_gradient.host_at_index(j) = (loss1 - loss2) / (2 * e);
          bias[i]->host_at_index(j) += e;
        }
        num_grads.push_back(numerical_gradient);
      }
      return num_grads;
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

    void init_network() { net_.setup(true); }

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
      net_.print_error();
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
    void test_onbatch(optimizer& opt, const tensor_t& in, const tensor_t& target, const size_t batch_size) {
      net_.setup(true);
      net_.forward_pass(in);
      net_.print_error();
      net_.backward(target);
      net_.update(opt, batch_size);
      net_.forward_pass(in);
      net_.print_error();
    };

    template <typename optimizer, typename OnBatchEnumerate, typename OnEpochEnumerate>
    bool train(optimizer& opt,
               const tensor_t& input,
               const tensor_t& train_labels,
               const size_t batch_size,
               const size_t epoch,
               OnBatchEnumerate on_batch_enumerate,
               OnEpochEnumerate on_epoch_enumerate,
               const std::string weight_and_bias_file,
               const std::string loss_filename,
               const std::string accuracy_filename,
               const bool reset_weight = false) {
      if (input.size() < train_labels.size()) {
        return false;
      }
      if (input.size() < batch_size || train_labels.size() < batch_size) {
        return false;
      }
      net_.setup(reset_weight);
      set_netphase(net_phase::train);
      opt.reset();
      stop_training_ = false;
      //time_t t       = clock();

      std::vector<float_t> loss;
      std::vector<float_t> accuracy;

      for (size_t i = 0; i < epoch && !stop_training_; ++i) {
        for (size_t j = 0; j < input.shape()[0] && !stop_training_; j += batch_size) {
          auto minibatch = input.subView({j}, {batch_size, input.dimension(dim_t::depth),
                                               input.dimension(dim_t::height), input.dimension(dim_t::width)});
          auto minilabels = train_labels.subView({j}, {batch_size, 1, 1, 1});
          train_once(opt, minibatch, minilabels, loss, accuracy);
          // on_batch_enumerate(t);
        }
        // on_epoch_enumerate(i);
      }

      save_to_file(weight_and_bias_file, content_type::weights_and_bias);
      save_data_to_file<float_t>(loss_filename, loss);
      save_data_to_file<float_t>(accuracy_filename, accuracy);

      return true;
    }

    void set_netphase(net_phase phase) {
      for (auto n : net_) {
        n->set_netphase(phase);
      }
    }

   private:
    /**
     * Trains on one minibatch, i.e. runs forward and backward propagation to
     * calculate
     * the gradient of the loss function with respect to the network parameters,
     * then calls the optimizer algorithm to update the weights
     *
     */
    template <typename optimizer>
    void train_once(optimizer& opt,
                    const tensor_t& minibatch,
                    const tensor_t& labels,
                    std::vector<float_t>& loss,
                    std::vector<float_t>& accuracy) {
      net_.forward_pass(minibatch);
      net_.backward_pass(labels, loss, accuracy);
      net_.update(opt);
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

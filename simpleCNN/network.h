//
// Created by hacht on 4/18/17.
//

#pragma once

#include "network_types.h"
#include "loss/loss_functions.h"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

namespace simpleCNN {

  typedef int label_t;

void display_filtermaps(const tensor_t& output, const size_t in_height, const size_t in_width) {
  cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE);
  cv::Mat image(in_height, in_width, CV_8UC1);

  for (size_t batch = 0; batch < output.shape()[0]; ++batch) {
    for (size_t filter = 0; filter < output.shape()[1]; ++filter) {

      uchar *p = image.data;
      for (size_t i = 0; i < in_height; ++i) {
        for (size_t j = 0; j < in_width; ++j) {
          auto val = output.host_at(batch, filter, i, j);
          p[i * in_width + j] = val;
        }
      }

      cv::imshow("Display window", image);
      cv::waitKey(0);
    }
  }
}

  template <typename NetType>
  class Network {
   public:
    explicit Network() : stop_training_(false) {}

    void gradient_check(const tensor_t& input, const tensor_t& labels, const size_t batch_size) {
      net_.setup(true);
      //net_.set_targets(labels);
      //auto ng  = computeNumericalGradient(input, labels);
      //printvt(ng, "Numerical gradient");

      //net_.forward_pass(input);
      tensor_t output = net_.forward(input);
      //print(output, "Output");
      tensor_t gradient = grads(output, labels);
      //print(gradient, "Gradients");
      net_.backward(gradient);
      //auto dW = net_.get_dW();
      //printvt_ptr(dW, "dW");

      //auto error = relative_error(dW, ng);
      //printvt(error, "Numerical error");
    }

    /**
     * Perturb each weight by +epsilon / -epsilon and compute loss.
     * Use this to approximate gradient.
     */
    std::vector<tensor_t> computeNumericalGradient(const tensor_t& input, const tensor_t& labels) {
      std::vector<tensor_t *> weights = net_.get_weights();
      size_t batch_size = input.shape()[0];
      std::vector<tensor_t> num_grads;

      float_t e = 1E-3;
      // For each layer containing weights.
      for (size_t i = 0; i < weights.size(); ++i) {
        std::cout << "Computing ... " + std::to_string(i) << std::endl;
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

    template<typename loss,typename optimizer>
    void test(optimizer& opt, const tensor_t& input, const tensor_t& labels) {
      net_.setup(true);
      train_once<loss>(opt, input, labels, input.shape()[0]);
    }

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
    void test_onbatch(
      optimizer& opt, const tensor_t& in, const tensor_t& target, const size_t batch_size) {
      net_.setup(true);
      net_.forward_pass(in);
      net_.print_error();
      net_.backward(target);
      net_.update(opt, batch_size);
      net_.forward_pass(in);
      net_.print_error();
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
      time_t t = clock();

      for (size_t i = 0; i < epoch && !stop_training_; ++i) {
        for (size_t j = 0; j < input.size() && !stop_training_; j += batch_size) {
          auto minibatch = input.subView({j},
                                         {batch_size, input.dimension(dim_t::depth), input.dimension(dim_t::height),
                                          input.dimension(dim_t::width)});
          //display_filtermaps(minibatch, 32, 32);
          auto minilabels = train_labels.subView({j}, {batch_size, 1, 1, 1});
          train_once<loss>(opt, minibatch, minilabels, batch_size);
          if (j % 1000 == 0) {
            loss_value(output_, minilabels);
            on_batch_enumerate(t);
          }
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
    tensor_t output_;
    tensor_t gradient_;
    /**
     * Trains on one minibatch, i.e. runs forward and backward propagation to
     * calculate
     * the gradient of the loss function with respect to the network parameters,
     * then calls the optimizer algorithm to update the weights
     *
     */
    template <typename loss, typename optimizer>
    void train_once(optimizer& opt, const tensor_t& minibatch, const tensor_t& labels, const size_t batch_size) {
      output_ = net_.forward(minibatch);
      gradient_ = grads(output_, labels);
      net_.backward(gradient_);
      net_.update(opt, batch_size);
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

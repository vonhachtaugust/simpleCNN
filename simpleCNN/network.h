//
// Created by hacht on 4/18/17.
//

#pragma once


#include <fstream>
#include "io/serialize.h"
#include "loss/loss_functions.h"
#include "network_types.h"
#include "optimizers/optimizer.h"
#include "util/math_functions.h"
#include "util/util.h"

namespace simpleCNN {

  typedef int label_t;

  enum class content_type { weights };

  enum class file_format { binary };

  template <typename NetType>
  class Network {
   public:
    explicit Network() : stop_training_(false) {}

    void save_results(std::vector<float_t>& mean_and_std, const std::string result) {
      std::string date = get_time_stamp();

      std::string tl_file = result + "/training_loss_" + date + ".txt";
      std::string vl_file = result + "/validation_loss_" + date + ".txt";
      std::string ta_file = result + "/training_accuracy_" + date + ".txt";
      std::string va_file = result + "/validation_accuracy_" + date + ".txt";
      std::string wf      = result + "/weights_" + date + ".txt";
      std::string mstd    = result + "/run_data_" + date + ".txt";
      std::string hyper   = result + "/hparas_dr_reg_lr_" + date + ".txt";

      save_content_to_file(wf, content_type::weights);
      save_data_to_file<float_t>(tl_file, training_loss_);
      save_data_to_file<float_t>(vl_file, validation_loss_);
      save_data_to_file<float_t>(ta_file, training_accuracy_);
      save_data_to_file<float_t>(va_file, validation_accuracy_);
      save_data_to_file<float_t>(mstd, mean_and_std);
    }

    void load_content_from_file(const std::string& filename, content_type what) const {
      std::ifstream ifs(filename.c_str(), std::ios::binary);

      if (ifs.fail() || ifs.bad()) {
        throw simple_error("Failed to open: " + filename);
      }

      from_archive(ifs, what);
    }

    void save_content_to_file(const std::string& filename, content_type what) const {
      std::ofstream ofs(filename.c_str(), std::ios::binary);

      if (ofs.fail() || ofs.bad()) {
        throw simple_error("Failed to open: " + filename);
      }

      to_archive(ofs, what);
    }

    template <typename OutputArchive>
    void to_archive(OutputArchive& ar, content_type what) const {
      if (what == content_type::weights) {
        net_.save_weights(ar);
      }
    }

    template <typename InputArchive>
    void from_archive(InputArchive& ar, content_type what) const {
      if (what == content_type::weights) {
        net_.load_weights(ar);
      }
    }

    void gradient_manual_check(const tensor_t& input, const tensor_t& labels, bool has_bias = true) {
      net_.setup(true);

      auto output = net_.forward_loss(input, labels);
      // print(output, "Output");
      net_.backward(labels);

      auto dW = net_.get_dW();
      // printvt_ptr(dW, "dW");

      if (has_bias) {
        auto dB = net_.get_dB();
        // printvt_ptr(dB, "dB");
      }
    }

    std::vector<float_t> gradient_check(const tensor_t& input, const tensor_t& labels, bool has_bias = true) {
      net_.setup(true);
      auto ng = computeNumericalGradient(input, labels);

      if (has_bias) {
        auto nb = computeNumericalGradient_bias(input, labels);
        // printvt(nb, "Numerical gradient dB");
      }

      auto output = net_.forward(input);
      // print(output, "output");
      net_.backward(labels);
      auto dW = net_.get_dW();

      if (has_bias) {
        auto dB = net_.get_dB();
        // printvt_ptr(dB, "dB");
      }

      // printvt(ng, "Numerical gradient dW");
      // printvt_ptr(dW, "dW");

      return relative_error(dW, ng);
      // return {1.0f};
    }

    std::vector<float_t> gradient_check_bias(const tensor_t& input, const tensor_t& labels) {
      net_.setup(true);
      auto ng = computeNumericalGradient_bias(input, labels);

      auto output = net_.forward(input);
      // print(output, "output");
      net_.backward(labels);
      auto dB = net_.get_dB();

      // printvt(ng, "Numerical gradient");
      // printvt_ptr(dB, "Backprop gradient");

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
      std::vector<tensor_t*> bias = net_.get_bias();
      size_t batch_size           = input.shape()[0];
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
    template <typename optimizer>
    void test_onbatch(optimizer& opt, const tensor_t& in, const tensor_t& target, const size_t batch_size) {
      net_.setup(true);
      auto error = net_.forward_loss(in, target);
      print(error, "First error");
      net_.backward(target);
      net_.update(opt, batch_size);
      auto output = net_.forward(in);
      print(output, "output");
    };

    template <typename optimizer, typename OnBatchEnumerate, typename OnEpochEnumerate>
    bool train(optimizer& opt,
               const tensor_t& training_images,
               const tensor_t& train_labels,
               const tensor_t& validation_images,
               const tensor_t& validation_labels,
               const size_t batch_size,
               const size_t epoch,
               OnBatchEnumerate on_batch_enumerate,
               OnEpochEnumerate on_epoch_enumerate,
               const bool store_result = true,
               const bool reset_weight = false) {
      if (training_images.size() < train_labels.size()) {
        return false;
      }
      if (training_images.size() < batch_size || train_labels.size() < batch_size) {
        return false;
      }
      net_.setup(reset_weight);
      opt.reset();
      stop_training_ = false;
      time_t t       = clock();

      std::vector<float_t> loss;
      std::vector<float_t> accuracy;

      for (size_t i = 0; i < epoch && !stop_training_; ++i) {
        for (size_t j = 0; j < training_images.shape()[0] && !stop_training_; j += batch_size) {
          if (float_t(training_images.shape()[0] - j) < batch_size) {
            continue;
          }

          size_t index = uniform_random(float_t(0), float_t(validation_images.shape()[0] - batch_size));
          auto minivi  = validation_images.subView(
            {index}, {batch_size, validation_images.dimension(dim_t::depth), validation_images.dimension(dim_t::height),
                      validation_images.dimension(dim_t::width)});
          auto minivl = validation_labels.subView({index}, {batch_size, 1, 1, 1});
          valid_once(minivi, minivl, store_result);

          auto minibatch = training_images.subView(
            {j}, {batch_size, training_images.dimension(dim_t::depth), training_images.dimension(dim_t::height),
                  training_images.dimension(dim_t::width)});
          auto minilabels = train_labels.subView({j}, {batch_size, 1, 1, 1});
          train_once(opt, minibatch, minilabels, store_result, batch_size);
        }
        on_batch_enumerate(t);
        on_epoch_enumerate(i);
      }
      return true;
    }

      void test_network(const tensor_t& test_images, const tensor_t& test_labels, const size_t batch_size, const size_t num_classes,
                        const std::string result, const bool display = false, const float_t mean = 0, const float_t std = 0) {
        size_t num_success = 0;
        size_t num_total   = 0;

        std::vector<float_t> confusion_matrix(num_classes * num_classes, 0.0);
        std::vector<float_t> accuracy;

        set_netphase(net_phase::test);
        size_t num_images = test_images.shape()[0];
        size_t image_size = test_images.size() / num_images;

        for (size_t j = 0; j < num_images; j += batch_size) {
          auto image =
              test_images.subView({j},
                                  {batch_size, test_images.dimension(dim_t::depth),
                                   test_images.dimension(dim_t::height),
                                   test_images.dimension(dim_t::width)});
          auto label = test_labels.subView({j}, {batch_size, 1, 1, 1});
          tensor_t output = net_.forward(image);
          size_t start_index = 0;

          for (size_t i = 0; i < batch_size; ++i) {
            tensor_t current_output = output.subView({i}, {1, 1, 10, 1});

            size_t max_index    = -1;
            float_t max         = float_t(-1);
            for (size_t n = 0; n < current_output.size(); ++n) {
              float_t value = *(current_output.host_begin() + n);
                if (value > max) {
                  max       = value;
                  max_index = n;
                }
            }

            size_t actual = *(label.host_begin() + i);
            size_t predicted = max_index;
#ifdef USE_OPENCV
            if (display) {
              auto currImage = image.subView({i}, {1, test_images.dimension(dim_t::depth),
                                                   test_images.dimension(dim_t::height),
                                                   test_images.dimension(dim_t::width)});
              display_image(currImage, actual, mean, std);
            }
#endif
            if (predicted == actual) {
              num_success++;
            }
            num_total++;
            confusion_matrix.at(actual * num_classes + predicted)++;
            start_index += num_classes;
          }
        }

        accuracy.push_back(float_t(num_success) / float_t(num_total));
        accuracy.push_back(float_t(num_total));

        std::string acc  = result + "/accuracy.txt";
        std::string conf = result + "/confusionmatrix.txt";

        save_data_to_file<float_t>(acc, accuracy);
        save_data_to_file<float_t>(conf, confusion_matrix);
      }

      void set_netphase(net_phase phase) {
        for (auto n : net_) {
          n->set_netphase(phase);
        }
      }

     private:
      void valid_once(const tensor_t& validation_batch, const tensor_t& validation_labels, const bool store_results) {
        set_netphase(net_phase::test);
        net_.forward_pass(validation_batch, validation_labels);
        net_.record_validation_progress(validation_loss_, validation_accuracy_, store_results);
      }

      /**
       * Trains on one minibatch, i.e. runs forward and backward propagation to
       * calculate
       * the gradient of the loss function with respect to the network parameters,
       * then calls the optimizer algorithm to update the weights
       *
       */
      template <typename optimizer>
      void train_once(optimizer & opt, const tensor_t& minibatch, const tensor_t& labels, const bool store_results,
                      const size_t batch_size) {
        set_netphase(net_phase::train);
        net_.forward_pass(minibatch, labels);
        net_.backward();
        net_.record_training_progress(training_loss_, training_accuracy_, store_results);
        net_.update(opt, batch_size);
      }

      template <typename layer>
      friend Network<Sequential>& operator<<(Network<Sequential>& n, layer&& l);

      NetType net_;
      net_phase phase_;
      bool stop_training_;

      std::vector<float_t> training_loss_;
      std::vector<float_t> validation_loss_;
      std::vector<float_t> training_accuracy_;
      std::vector<float_t> validation_accuracy_;
    };

    template <typename layer>
    Network<Sequential>& operator<<(Network<Sequential>& n, layer&& l) {
      n.net_.add(std::forward<layer>(l));
      return n;
    }
}  // namespace simpleCNN

//
// Created by hacht on 4/18/17.
//

#pragma once

#include "layers/layer.h"
#include "optimizers/optimizer.h"

namespace simpleCNN {
  class Network_type {
   public:
    typedef std::vector<Layer*>::iterator iterator;
    typedef std::vector<Layer*>::const_iterator const_iterator;

    virtual tensor_t forward(const tensor_t& input) = 0;

    virtual void forward_pass(const tensor_t& input) = 0;

    virtual float_t forward_loss(const tensor_t& input, const tensor_t& labels) = 0;

    virtual void backward(const tensor_t& labels) = 0;

    virtual void update(Optimizer& opt) {
      for (auto l : nodes_) {
        l->update(opt);
      }
    }

    virtual void setup(bool reset_weight) {
      for (auto l : nodes_) {
        l->setup(reset_weight);
      }
    }

    std::vector<tensor_t*> get_dW() {
      std::vector<tensor_t *> dW;
      for (auto l : nodes_) {
        l->get_dW(dW);
      }

      if (dW.size() == 0) {
        throw simple_error("Network has no trainable weights");
      }

      return dW;
    }

    void print_dW() {
      for (auto l : nodes_) {
        l->print_dW();
      }
    }

    virtual std::vector<tensor_t*> get_bias() {
      std::vector<tensor_t *> bias;

      for (auto l : nodes_) {
        l->get_bias(bias);
      }
      return bias;
    }

    virtual std::vector<tensor_t*> get_weights() {
      std::vector<tensor_t*> weights;

      for (auto l : nodes_) {
        l->get_weights(weights);
      }
      return weights;
    }

    void print_error() {
      std::cout << nodes_.back()->error() << std::endl;
    }

    template<typename OutputArchive>
    void save_weight_and_bias(OutputArchive &oa) const {
      for (auto l : nodes_) {
        l->save(oa);
      }
    }

    template<typename InputArchive>
    void load_weight_and_bias(InputArchive &ia) const {
      for (auto l : nodes_) {
        l->load(ia);
      }
    }

    size_t size() const { return nodes_.size(); }
    iterator begin() { return nodes_.begin(); }
    iterator end() { return nodes_.end(); }
    const_iterator begin() const { return nodes_.begin(); }
    const_iterator end() const { return nodes_.end(); }
    Layer* operator[](size_t i) { return nodes_[i]; }
    const Layer* operator[](size_t i) const { return nodes_[i]; }

   protected:
    /**
     * Allocates memory for this layer.                        -- /
     *
     */
    template <typename T>
    void push_back(T&& layer) {
      push_back_impl(std::forward<T>(layer), typename std::is_rvalue_reference<decltype(layer)>::type());
    }

    template <typename T>
    void push_back_impl(T&& layer, std::true_type) {
      own_nodes_.push_back(std::make_shared<typename std::remove_reference<T>::type>(std::forward<T>(layer)));
      nodes_.push_back(own_nodes_.back().get());
    }

    template <typename T>
    void push_back_impl(T&& layer, std::false_type) {
      nodes_.push_back(&layer);
    }

    /** --                                                    -- */
    std::vector<std::shared_ptr<Layer>> own_nodes_;
    std::vector<Layer*> nodes_;
  };

  class Sequential : public Network_type {
   public:
    void backward(const tensor_t& labels) override {
      nodes_.back()->set_targets(labels);

      for (auto l = nodes_.rbegin(); l != nodes_.rend(); ++l) {
        (*l)->backward();
      }
    }

    void backward_pass(const tensor_t& labels, std::vector<float_t>& loss, std::vector<float_t>& accuracy) {
      nodes_.back()->set_targets(labels);

      for (auto l = nodes_.rbegin(); l != nodes_.rend(); ++l) {
        (*l)->backward();
      }

      auto err = nodes_.back()->error();
      loss.push_back(err);

      auto acc = nodes_.back()->accuracy();
      accuracy.push_back(acc);
    }

    tensor_t forward(const tensor_t& input) override {
      nodes_.front()->set_in_data(input, component_t::IN_DATA);

      for (auto l : nodes_) {
        l->forward();
      }

      return nodes_.back()->output();
    }

    float_t forward_loss(const tensor_t& input, const tensor_t& labels) override {
      nodes_.front()->set_in_data(input, component_t::IN_DATA);

      for (size_t i = 0; i < nodes_.size(); ++i) {
        nodes_[i]->forward();
      }

      return nodes_.back()->error();
    }

    void forward_pass(const tensor_t& input) override {
      nodes_.front()->set_in_data(input, component_t::IN_DATA);

      for (auto l : nodes_) {
        l->forward();
      }
    }

    template <typename T>
    void add(T&& layer) {
      push_back(std::forward<T>(layer));

      if (size() != 1) {
        auto head = nodes_[size() - 2];
        auto tail = nodes_[size() - 1];
        head->connect(tail);
        check_connectivity();
      }
    }

    void check_connectivity() {
      for (size_t i = 0; i < size() - 1; ++i) {
        auto out = nodes_[i]->outputs();
        auto in  = nodes_[i + 1]->inputs();

        if (out[0] != in[0]) {
          throw simple_error("Connection failure");
        }
      }
    }
  };
}  // namespace simpleCNN

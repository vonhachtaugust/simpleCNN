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

    virtual void backward(const tensor_t& deltas) = 0;

    virtual void update_weights(Optimizer& opt) {
      for (auto l : nodes_) {
        l->update_weight(opt);
      }
    }

    virtual void setup(bool reset_weight) {
      for (auto l : nodes_) {
        l->setup(reset_weight);
      }
    }

    size_t size() const { return nodes_.size(); }
    iterator begin() { nodes_.begin(); }
    iterator end() { nodes_.end(); }
    const_iterator begin() const { nodes_.begin(); }
    const_iterator end() const { nodes_.end(); }
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
    void backward(const tensor_t& deltas) override {
      nodes_.back()->set_out_grads(deltas, component_t::OUT_GRAD);
      size_t n = nodes_.size();

      for (size_t i = 0; i < n; ++i) {
        nodes_[n - 1 - i]->backward();
      }
    }

    tensor_t forward(const tensor_t& input) override {
      nodes_.front()->set_in_data(input, component_t::IN_DATA);

      for (size_t i = 0; i < nodes_.size(); ++i) {
        nodes_[i]->forward();
      }
      return nodes_.back()->output();
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
      for (size_t i = 0; i < size(); ++i) {
        auto out = nodes_[i]->outputs();
        auto in  = nodes_[i + 1]->inputs();

        if (out[0] != in[0]) {
          throw simple_error("Connection failure");
        }
      }
    }
  };

}  // namespace simpleCNN
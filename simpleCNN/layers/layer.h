//
// Created by hacht on 3/4/17.
//

#pragma once

#include <iomanip>
#include <sstream>
#include "../core/backend.h"
#include "../core/framework/device.h"
#include "../node.h"
#include "../optimizers/optimizer.h"
#include "../util/util.h"
#include "../util/weight_init.h"
#include "../util/math_functions.h"
#include "../network.h"

namespace simpleCNN {
  /**
   * base class of all kind of NN layers
  *
  * sub-class should override these methods:
  * - forward_propagation ... forward pass calculation
  * - back_propagation    ... backward pass calculation - deltas
  * - in_shape            ... specify input data shapes
  * - out_shape           ... specify output data shapes
  * - layer_type          ... name of layer
  **/
  class Layer : public Node {
   public:
    /**
    * @brief Default layer constructor that instantiates a N-input, M-output
    *layer
    *
    * @param in_type[N] type of input vector (data, weight, bias...)
    * @param out_type[M] type of output vector
    *
    **/
    Layer(const data_t &in_type, const data_t &out_type)
        : Node(static_cast<size_t>(in_type.size()), static_cast<size_t>(out_type.size())),
          initialized_(false),
          in_channels_(static_cast<size_t>(in_type.size())),
          out_channels_(static_cast<size_t>(out_type.size())),
          in_type_(in_type),
          out_type_(out_type) {
      weight_init_ = std::make_shared<weight_init::Gaussian>();
      bias_init_ = std::make_shared<weight_init::Constant>();
    }

    virtual ~Layer() = default;

    /*
     * The implicitly-defined copy/move constructor for a non-union class X
     * performs activate memberwise copy/move of its bases and members.
     */
    Layer(const Layer &) = default;
    Layer &operator=(const Layer &) = default;

    /*
     * Move constructors
     */
    // Layer(Layer&&) = default;
    // Layer &operator=(const Layer&&) = default;

    Layer &set_backend_type(core::backend_t backend_type) {
      backend_type_ = backend_type;
      return *this;
    }

    /** Start: Getters -------------------------------------- */
    core::backend_t engine() const { return backend_type_; }

    Device *device() const { return device_ptr_.get(); }

    // number of incoming edges in this layer
    size_t in_channels() const { return in_channels_; }

    // number of outcoming edge in this layer
    size_t out_channels() const { return out_channels_; }

    data_t in_type() const { return in_type_; }

    data_t out_type() const { return out_type_; }

    void set_trainable(bool trainable) { trainable_ = trainable; }

    bool trainable() const { return trainable_; }

    tensor_t *in_component(component_t t) {
      for (size_t i = 0; i < in_channels_; ++i) {
        if (in_type_[i].getComponentType() == t) {
          return get_component_data(i, t);
        }
      }
      throw simple_error("Error: In component not allocated.");
    }

    tensor_t *out_component(component_t t) {
      for (size_t i = 0; i < out_channels_; ++i) {
        if (out_type_[i].getComponentType() == t) {
          return ith_out_node(i)->get_data();
        }
      }
      throw simple_error("Error: Out component not allocated.");
    }

    /** End: Getters ---------------------------------------- */

    /** Start: Setters -------------------------------------- */
    Layer &set_device(Device *device) {
      device_ptr_.reset(device);
      return *this;
    }

    virtual void set_in_shape(const shape4d& shape) {
      throw simple_not_implemented_error();
    }

    template<typename WeightInit>
    Layer &weight_init(const WeightInit &f) {
      weight_init_ = std::make_shared<WeightInit>(f);
      return *this;
    }

    template<typename BiasInit>
    Layer &bias_init(const BiasInit &f) {
      bias_init_ = std::make_shared<BiasInit>(f);
      return *this;
    }

    void set_in_data(const tensor_t &data, component_t ct) { *in_component(ct) = data; }

    void set_out_data(const tensor_t &data, component_t ct) { *out_component(ct) = data; }

    void set_out_grads(const tensor_t &delta, component_t ct) {
      *ith_out_node(0)->get_gradient() = delta;
    }

    /** End: Setters ---------------------------------------- */

    std::vector<edgeptr_t> inputs() {
      std::vector<edgeptr_t> nodes(in_channels_);
      for (size_t i = 0; i < in_channels_; ++i) {
        nodes[i] = ith_in_node(i);
      }
      return nodes;
    }

    std::vector<edgeptr_t> outputs() {
      std::vector<edgeptr_t> nodes(out_channels_);
      for (size_t i = 0; i < out_channels_; ++i) {
        nodes[i] = ith_out_node(i);
      }
      return nodes;
    }

    tensor_t output() { return *ith_out_node(0)->get_data(); }

    void setup(bool reset_weight) {
      /**
       * Verifies that in_shape (called from derived layer class) suits the
       * required number of input channels. Does the same for out_shape.
       */
      if (in_shape().size() != in_channels_ || out_shape().size() != out_channels_) {
        throw simple_error("Connection mismatch at layer setup");
      }

      /**
       * Allocates memory for the output data.
       */
      for (size_t i = 0; i < out_channels_; ++i) {
        if (!next_[i]) {
          next_[i] = std::make_shared<Edge>(this, out_shape()[i]);
        }
      }

      /**
       * Allocates memory for weight and bias.
       */
      if (reset_weight || !initialized_) {
        init_weight();
      }
    }

    void clear_gradients() {
      for (size_t i = 0; i < in_channels_; ++i) {
        ith_in_node(i)->clear_gradients();
      }
    }

    void print_gradients() {
      if (trainable()) {
        for (size_t i = 0; i < in_channels_; ++i) {
          auto type = in_type_[i].getComponentType();
          if (type == component_t::WEIGHT) {
            auto W = get_component_data(i, type);
            auto dW = get_component_gradient(i, type);
            print(*W, "W");
            print(*dW, "dW");
          }
        }
      }
    }

    void update(Optimizer &opt, const size_t batch_size) {
      if (trainable()) {
        for (size_t i = 0; i < in_channels_; ++i) {
          auto type = in_type_[i].getComponentType();
          if (type == component_t::WEIGHT) {
            auto W = get_component_data(i, type);
            auto dW = get_component_gradient(i, type);

            classifier_ ? mean_and_regularize(*W, *dW, batch_size) : mean(*dW, batch_size);
            opt.update(dW, W);
          }
          if (type == component_t::BIAS) {
            auto b = get_component_data(i, type);
            auto db = get_component_gradient(i, type);

            mean(*db, batch_size);
            opt.update(db, b);
          }
        }
      }
    }

    /**
 * @brief Initalizes the vectors containing the trainable data
 */
    void init_weight() {
      if (!trainable_) {
        initialized_ = true;
        return;
      }

      for (size_t i = 0; i < in_channels_; i++) {
        component_t type_ = in_type_[i].getComponentType();
        switch (type_) {
          case component_t::WEIGHT:weight_init_->fill(get_component_data(i, type_), fan_in_size(), fan_out_size());
            break;
          case component_t::BIAS: bias_init_->fill(get_component_data(i, type_), fan_in_size(), fan_out_size());
            break;
          default: break;
        }
      }
      initialized_ = true;
    }

    tensor_t* gradients(component_t component) {
      for (size_t i = 0; i < in_channels_; ++i) {
        component_t type = in_type_[i].getComponentType();
        if (type == component) {
          return get_component_gradient(i, type);
        }
      }
    }

    void print_layer_data() {
      print(layer_type(), "Layer type");

      for (size_t i = 0; i < in_channels_; ++i) {
        const auto &in = ith_in_node(i);
        print(*in->get_data(), "Input data " + std::to_string(i));
        print(*in->get_gradient(), "Input grad " + std::to_string(i));
      }

      for (size_t i = 0; i < out_channels_; ++i) {
        const auto &out = ith_out_node(i);
        print(*out->get_data(), "Output data " + std::to_string(i));
        print(*out->get_gradient(), "Output grad " + std::to_string(i));
      }
    }

    void forward() {
      data_ptrs_t in_data(in_channels_), out_data(out_channels_);

      for (size_t i = 0; i < in_channels_; ++i) {
        in_data[i] = ith_in_node(i)->get_data();
      }

      for (size_t i = 0; i < out_channels_; ++i) {
        out_data[i] = ith_out_node(i)->get_data();
        ith_out_node(i)->clear_gradients();
      }

      forward_propagation(in_data, out_data);
    }

    void backward() {
      data_ptrs_t in_data(in_channels_), in_grad(in_channels_), out_data(out_channels_), out_grad(out_channels_);

      for (size_t i = 0; i < in_channels_; ++i) {
        const auto &in = ith_in_node(i);
        in_data[i] = in->get_data();
        in_grad[i] = in->get_gradient();
      }

      for (size_t i = 0; i < out_channels_; ++i) {
        const auto &out = ith_out_node(i);
        out_data[i] = out->get_data();
        out_grad[i] = out->get_gradient();
      }

      back_propagation(in_data, out_data, in_grad, out_grad);
    }

    /** Start: Virtuals ------------------------------------- */

     /**
    * return output value range used only for calculating target
    * value from label-id in final(output) layer override properly
    * if the layer is intended to be used as output layer
    **/
    virtual std::pair<float_t, float_t> out_value_range() const { return {float_t{0.0}, float_t{1.0}}; }

    virtual void set_netphase(net_phase phase) {}

    /**
    * number of incoming connections for each output unit
    * used only for weight/bias initialization methods which require fan-in
    * size (e.g. xavier) override if the layer has trainable weights, and
    * scale of initialization is important.
    **/
    virtual size_t fan_in_size() const { return *(in_shape()[0].end() - 1); }

    /**
    * number of outgoing connections for each input unit
    * used only for weight/bias initialization methods which require fan-out
    * size (e.g. xavier) override if the layer has trainable weights, and
    * scale of initialization is important
    **/
    virtual size_t fan_out_size() const { return *(out_shape()[0].end() - 1); }

    /** End: Virtuals ---------------------------------------- */

    /** Start: Pure virtuals --------------------------------- */

    /**
    * array of input shapes
    *
    **/
    virtual shape_t in_shape() const = 0;

    /**
    * array of output shapes (width x height x depth)
    **/
    virtual shape_t out_shape() const = 0;

    /**
    * name of layer, should be unique for each concrete class
    **/
    virtual std::string layer_type() const = 0;

    /**
     * Performs a forward propagation and computes the input of the next layer
     *
     * @param in_data       input vector of this layer (in_data, weight, bias)
     * @param out_data      output vector (out_data)
     */
    virtual void forward_propagation(const data_ptrs_t &in_data, data_ptrs_t &out_data) = 0;

    /**
     * Performs a backward proagation and 're-computes' the output from the previous layer
     *
     * @param in_data       input data vector of this layer (in_data, weight, bias)
     * @param out_data      output data vector (out_data)
     * @param in_grad       input grad vector of this layer (prev_grad, dW, db)
     * @param out_grad      output data vector (curr_grad)
     */
    virtual void back_propagation(const data_ptrs_t &in_data,
                                  const data_ptrs_t &out_data,
                                  data_ptrs_t &in_grad,
                                  data_ptrs_t &out_grad) = 0;

    /** End: Pure virtuals ----------------------------------- */

    void reshape(tensor_t &data, const shape4d &shape) {
      bool re = false;

      for (size_t i = 0; i < shape.size(); ++i) {
        if (data.shape()[i] != shape[i]) {
          re = true;
          break;
        }
      }

      if (re) {
        data.reshape(shape);
      }
    }

    inline void connect(Layer* next) {
      auto out_shape = this->out_shape()[0];
      auto in_shape = next->in_shape()[0];

      this->setup(false);

      /** Activation layer shape is equal to previous layer shape */
      if (in_shape.size() == 0) {
        next->set_in_shape(out_shape);
        in_shape = out_shape;
      }

      if (out_shape.size() != in_shape.size()) {
        throw simple_error("Connection mismatch");
      }

      if (!this->next_[0]) {
        throw simple_error("Output edge must not be null");
      }

      next->prev_[0] = this->next_[0];
      next->prev_[0]->add_next_node(next);
    }
    
    const void set_as_classifier() { classifier_ = true; }

   protected:
    /**
     * Flag to indicate whether the layer is a classifier layer.
     */
    bool classifier_ = false;
    
    /**
     * Flag indication whether the layer/node is initialized
     */
    bool initialized_;

    /**
     * The number of input vectors/edges
     */
    size_t in_channels_;

    /**
     * The number of output vectors/edges
     */
    size_t out_channels_;

    /**
     * Vector containing the type of data for inputs
     */
    data_t in_type_;

    /**
     * Vector containing the type of data for outputs
     */
    data_t out_type_;

    /**
     * The current backend type for operations
     */
    core::backend_t backend_type_;

    /**
     * Pointer to the device on which the layer/node will run
     */
    std::shared_ptr<Device> device_ptr_ = nullptr;

    private:
    /**
     * @brief Allocates the necessary edge memory in a specific incoming connection.
     */
    void alloc_input(size_t i) const { prev_[i] = std::make_shared<Edge>(nullptr, in_shape()[i]); }

    /**
     * @brief Allocates the necessary edge memory in a specific outcoming connection.
     */
    void alloc_output(size_t i) const { next_[i] = std::make_shared<Edge>(const_cast<Layer*>(this), out_shape()[i]); }

    /**
     * @brief Creates an edge between the current node and one incoming or previous node.
     */
    edgeptr_t ith_in_node(size_t i) {
      if (!prev_[i]) {
        alloc_input(i);
      }
      return prev()[i];
    }

    /**
     * @brief Creates an edge between the current node and one outcoming or next node.
     */
    edgeptr_t ith_out_node(size_t i) {
      if (!next_[i]) {
        alloc_output(i);
      }
      return next()[i];
    }

    /**
     * @brief Retrieves weight tensor from incoming edge
     */
    tensor_t* get_component_data(size_t i, component_t t) {
      assert(in_type_[i].getComponentType() == t);
      return ith_in_node(i)->get_data();
    }

    tensor_t* get_component_gradient(size_t i, component_t t) {
      assert(in_type_[i].getComponentType() == t);
      return ith_in_node(i)->get_gradient();
    }

    /**
     * Flag indicating whether the layer/node parameters are trainable
     */
    bool trainable_;

    /**
     * Pointer to the function for weights initialization
     */
    std::shared_ptr<weight_init::Function> weight_init_;

    /**
     * Pointer to the function for biases initialization
     */
    std::shared_ptr<weight_init::Function> bias_init_;
  };

}  // namespace simpleCNN

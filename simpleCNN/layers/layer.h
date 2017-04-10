//
// Created by hacht on 3/4/17.
//

#pragma once

#include "../core/backend.h"
#include "../core/framework/device.h"
#include "../node.h"
#include "../util/util.h"
#include "../util/weight_init.h"

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
    Layer(const data_t& in_type, const data_t& out_type)
      : Node(static_cast<size_t>(in_type.size()), static_cast<size_t>(out_type.size())),
        initialized_(false),
        in_channels_(static_cast<size_t>(in_type.size())),
        out_channels_(static_cast<size_t>(out_type.size())),
        in_type_(in_type),
        out_type_(out_type) {
      weight_init_ = std::make_shared<weight_init::Xavier>();
      bias_init_   = std::make_shared<weight_init::Constant>();
      trainable_   = true;
    }

    virtual ~Layer() = default;

    /*
     * The implicitly-defined copy/move constructor for a non-union class X
     * performs a memberwise copy/move of its bases and members.
     */
    Layer(const Layer&) = default;
    Layer& operator=(const Layer&) = default;

    /*
     * Move constructors
     */
    // Layer(Layer&&) = default;
    // Layer &operator=(const Layer&&) = default;

    Layer& set_backend_type(core::backend_t backend_type) {
      backend_type_ = backend_type;
      return *this;
    }

    // Start: Getters -------------------------------------- //
    core::backend_t engine() const { return backend_type_; }

    Device* device() const { return device_ptr_.get(); }

    // number of incoming edges in this layer
    size_t in_channels() const { return in_channels_; }

    // number of outcoming edge in this layer
    size_t out_channels() const { return out_channels_; }

    // in_type_[0]: data, in_type_[1]: weights, in_type_[2]: bias
    // tensor_t weights() const { return in_type_[1]; }

    // tensor_t weights() { return in_type_[1]; }

    data_t in_type() const { return in_type_; }

    data_t out_type() const { return out_type_; }

    void set_trainable(bool trainable) { trainable_ = trainable; }

    bool trainable() const { return trainable_; }

    tensor_t* in_component(component_t t) {
      for (size_t i = 0; i < in_channels_; ++i) {
        if (in_type_[i].getComponentType() == t) {
          return get_component_data(i, t);
        }
      }
      throw simple_error("Error: In component not allocated.");
    }

    tensor_t* out_component(component_t t) {
      for (size_t i = 0; i < out_channels_; ++i) {
        if (out_type_[i].getComponentType() == t) {
          return ith_out_node(i)->get_data();
        }
      }
      throw simple_error("Error: Out component not allocated.");
    }

    // End: Getters ---------------------------------------- //

    // Start: Setters -------------------------------------- //
    Layer& set_device(Device* device) {
      device_ptr_.reset(device);
      return *this;
    }

    template <typename WeightInit>
    Layer& weight_init(const WeightInit& f) {
      weight_init_ = std::make_shared<WeightInit>(f);
      return *this;
    }

    template <typename BiasInit>
    Layer& bias_init(const BiasInit& f) {
      bias_init_ = std::make_shared<BiasInit>(f);
      return *this;
    }

    void set_in_data(const tensor_t& data, component_t ct) { *in_component(ct) = data; }

    void set_out_data(const tensor_t& data, component_t ct) { *out_component(ct) = data; }

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
          case component_t::WEIGHT:
            weight_init_->fill(get_component_data(i, type_), fan_in_size(), fan_out_size());
            break;
          case component_t::BIAS: bias_init_->fill(get_component_data(i, type_), fan_in_size(), fan_out_size()); break;
          default: break;
        }
      }

      initialized_ = true;
    }

    void forward() {
      tensor_t in;
      tensor_t out;
      //forward_propagation({}, {});
      forward_activation(in, out);
    }

    void backward() {
      tensor_t in;
      tensor_t out;
      //back_propagation({}, {}, {}, {});
      backward_activation(in, in, out);
    }

    // End: Setters ---------------------------------------- //

    // Start: Virtuals ------------------------------------- //
    /**
    * return output value range used only for calculating target
    * value from label-id in final(output) layer override properly
    * if the layer is intended to be used as output layer
    **/
    virtual std::pair<float_t, float_t> out_value_range() const { return {float_t{0.0}, float_t{1.0}}; }

    virtual void createOp() {}

    // End: Virtuals ---------------------------------------- //

    // Start: Pure virtuals --------------------------------- //
    /**
    * array of input shapes (height x width x depth)
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

    virtual void forward_propagation(const data_ptrs_t& in_data, data_ptrs_t& out_data) = 0;

    virtual void back_propagation(
            const data_ptrs_t & in_data,
            const data_ptrs_t & out_data,
            data_ptrs_t & in_grad,
            data_ptrs_t & out_grad) = 0;

    virtual void forward_activation(const tensor_t& affine, tensor_t& activated) = 0;

    virtual void backward_activation(const tensor_t& prev_delta, const tensor_t& affine, tensor_t& activated) = 0;

    // End: Pure virtuals ----------------------------------- //

   protected:
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

    /** Used in update_weight method. Kept as a member variable to reduce
     * frequent memory allocation
     */
    vec_t weights_diff_;

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

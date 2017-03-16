//
// Created by hacht on 3/4/17.
//

#pragma once

#include "../core/backend.h"
#include "../core/framework/device.h"
#include "../node.h"
#include "../util/util.h"

namespace simpleCNN {
  /**
   * base class of all kind of NN layers
  *
  * sub-class should override these methods:
  * - forward_propagation ... body of forward-pass calculation
  * - back_propagation    ... body of backward-pass calculation
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
      : Node(static_cast<size_t>(in_type.size()),
             static_cast<size_t>(out_type.size())),
        initialized_(false),
        in_channels_(static_cast<size_t>(in_type.size())),
        out_channels_(static_cast<size_t>(out_type.size())),
        in_type_(in_type),
        out_type_(out_type) {
      //            weight_init_ = std::make_shared<weight_init::xavier>();
      //            bias_init_   = std::make_shared<weight_init::constant>();
      trainable_ = true;
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
    tensor_t weights() const { return in_type_[1]; }

    tensor_t weights() { return in_type_[1]; }

    data_t in_type() const { return in_type_; }

    data_t out_type() const { return out_type_; }

    void set_trainable(bool trainable) { trainable_ = trainable; }

    bool trainable() const { return trainable_; }

    // End: Getters ---------------------------------------- //

    // Start: Setters -------------------------------------- //
    Layer& set_device(Device* device) {
      device_ptr_.reset(device);
      return *this;
    }

    // End: Setters ---------------------------------------- //

    // Start: Virtuals ------------------------------------- //
    /**
    * return output value range used only for calculating target
    * value from label-id in final(output) layer override properly
    * if the layer is intended to be used as output layer
    **/
    virtual std::pair<float_t, float_t> out_value_range() const {
      return {float_t{0.0}, float_t{1.0}};
    }

    virtual void createOp() {}
    // End: Virtuals ---------------------------------------- //

    // Start: Pure virtuals --------------------------------- //
    /**
    * array of input shapes (height x width x depth)
    *
    **/
    virtual data_t in_shape() const = 0;

    /**
    * array of output shapes (width x height x depth)
    **/
    virtual data_t out_shape() const = 0;

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
    virtual size_t fan_in_size() const { return in_shape()[0].shape().at(0); }

    /**
    * number of outgoing connections for each input unit
    * used only for weight/bias initialization methods which require fan-out
    * size (e.g. xavier) override if the layer has trainable weights, and
    * scale of initialization is important
    **/
    virtual size_t fan_out_size() const { return out_shape()[0].shape().at(0); }

    /////////////////////////////////////////////////////////////////////////
    // fprop/bprop

    /**
    * @param in_data  input vectors of this layer (data, weight, bias)
    * @param out_data output vectors
    **/
    virtual void forward_propagation(const vec_tensor_ptr_t& in_data,
                                     vec_tensor_ptr_t& out_data) = 0;

    /**
    * return delta of previous layer (delta=\frac{dE}{da}, a=wx in
    *fully-connected layer)
    * @param in_data  input vectors (same vectors as forward_propagation)
    * @param out_data output vectors (same vectors as forward_propagation)
    * @param out_grad gradient of output vectors (i-th vector correspond with
    *out_data[i])
    * @param in_grad  gradient of input vectors (i-th vector correspond with
    *in_data[i])
    **/
    /* virtual void back_propagation(
            const vec_tensor_ptr_t& in_data,
            const vec_tensor_ptr_t& out_data,
            vec_tensor_ptr_t& out_grad,
            vec_tensor_ptr_t& in_grad) = 0; */

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
     * Flag indicating whether the layer/node parameters are trainable
     */
    bool trainable_;

    /**
     * Pointer to the function for weights initialization
     */
    //        std::shared_ptr<weight_init::function> weight_init_;

    /**
     * Pointer to the function for biases initialization
     */
    //        std::shared_ptr<weight_init::function> bias_init_;
  };
}  // namespace simpleCNN

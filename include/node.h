//
// Created by hacht on 3/4/17.
//

#pragma once

#include <memory>
#include <vector>
#include "util/util.h"

namespace simpleCNN {
    class Edge;

    class Node;

    typedef std::shared_ptr<Node> nodeptr_t;
    typedef std::shared_ptr<Edge> edgeptr_t;

    class Node : public std::enable_shared_from_this<Node> {
    public:
        explicit Node(
                size_t in_size,
                size_t out_size
        )
                :prev_(in_size), next_(out_size)
        {
        }

        virtual ~Node() { }

        /*
         * View list of ingoing connections (edges)
         */
        const std::vector<edgeptr_t>& prev() const
        {
            return prev_;
        }

        /*
         * View list of outgoing connections (edges)
         */
        const std::vector<edgeptr_t>& next() const
        {
            return next_;
        }

    protected:
        Node() = delete;

        /*
         * Connection verification
         */
        /* virtual void connect(
                nodeptr_t head,
                nodeptr_t tail,
                size_t head_index,
                size_t tail_index) = 0; */

        /*
         * Vector of edges going out and in of this node.
         *
         * Edge ex. Weights i.e an edge holds the weights data
         */
        mutable std::vector<edgeptr_t> prev_;
        mutable std::vector<edgeptr_t> next_;
    };

    /**
    * class containing input/output data
    **/
    class Edge {
    public:
        Edge(nodeptr_t prev,
                const std::initializer_list<size_t>& shape
        )
                :shape_(shape),
                 data_(shape),
                 grad_(shape),
                 prev_(prev) { }

        // Getter: ----------------------------------------------- //
        tensor_t& get_data() { return data_; }

        const tensor_t& get_data() const { return data_; }

        /*
         * Gradient data for weights
         */
        tensor_t& get_gradient() { return grad_; }

        const tensor_t& get_gradient() const { return grad_; }

        /*
         * Next node to which this edge connect to
         */
        nodeptr_t& next() { return next_; }

        const nodeptr_t& next() const { return next_; }

        /*
         * Previous node to which this edge connect from
         */
        nodeptr_t& prev() { return prev_; }

        const nodeptr_t& prev() const { return prev_; }

        /*
         * Width x Height x Depth description of tensor.
         */
        const std::initializer_list<size_t>& shape() const { return shape_; }
        // ------------------------------------------------------- //

        Edge& add_next_node(nodeptr_t next)
        {
            next_ = next;
            return *this;
        }

    private:
        std::initializer_list<size_t> shape_;
        tensor_t data_;
        tensor_t grad_;
        nodeptr_t prev_;
        nodeptr_t next_;
    };
} // namespace simpleCNN

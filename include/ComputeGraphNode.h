#ifndef __ICDL_COMPUTE_GRAPH_NODE_H__
#define __ICDL_COMPUTE_GRAPH_NODE_H__
#include "Tensor.h"
#include <memory>
#include "Operator.h"
#include <type_traits>
namespace icdl{
    class ComputeGraph;
    class DynamicComputeGraph;
    enum class ComputeGraphNodeType{
        COMPUTE_GRAPH,
        OPERATOR,
        TENSOR, //not used now
        INVALID_NTYPE
    };

    class ComputeGraphNode{
    private:
        ComputeGraphNodeType _ntype{ComputeGraphNodeType::INVALID_NTYPE};
    protected:
        void set_node_type(const ComputeGraphNodeType& ntype){
            _ntype = ntype;
        }
    public:
        explicit ComputeGraphNode(const ComputeGraphNodeType& ntype): _ntype(ntype){}
        ComputeGraphNode(){}
        ComputeGraphNodeType get_node_type() const;
    };

    class ComputeNode: public ComputeGraphNode{
    private:
        std::shared_ptr<Operator> _op_ptr{nullptr};
        std::shared_ptr<DynamicComputeGraph> _sub_graph_ptr{nullptr};
    public:
        // ComputeNode node(icdl::LinearOpMake(3,10, makeLinearPytorchImpl()));
        // the LinearOpMake may return a shared_ptr<Linear>, but we use this to initialize the Operator ptr.
        /*
        template<typename OP_OR_GRAPH_TYPE>
        explicit ComputeNode(std::shared_ptr<OP_OR_GRAPH_TYPE> ptr);
        */
        
        ComputeNode(std::shared_ptr<Operator> ptr)
            : ComputeGraphNode(ComputeGraphNodeType::OPERATOR), _op_ptr(ptr){}
        ComputeNode(std::shared_ptr<DynamicComputeGraph> ptr)
            : ComputeGraphNode(ComputeGraphNodeType::COMPUTE_GRAPH), _sub_graph_ptr(ptr){}
        // make an invalid node.
        ComputeNode(){}
        // ComputeNode node(icdl::SeqGraphMake(op_list))
        /*
        explicit ComputeNode(std::shared_ptr<ComputeGraph> sub_graph_ptr)
            : ComputeGraphNode(ComputeGraphNodeType::COMPUTE_GRAPH), _sub_graph_ptr(sub_graph_ptr){}
        */
        /*
        // https://www.zhihu.com/question/67429387
        // templates only for derived types of ComputeGraph
        // example:
        // SeqGraph graph;
        // graph.add_compute_node("conv1", icdl::Conv2dOpMake(3,3,3, makeConv2dPytorchImpl()));
        // graph.add_compute_node("fc1", icdl::LinearOpMake(3,10, makeLinearPytorchImpl()));
        // ComputeNode node(graph); //a node contains a sub_graph
        // SeqGraph g2(op_list);
        // ComputeNode node2(g2);
        template<typename ComputeGraphType, typename std::enable_if<std::is_base_of<ComputeGraph, ComputeGraphType>{}, int>::type=0>
        explicit ComputeNode(ComputeGraphType& sub_graph);
        */
        std::shared_ptr<Operator> get_op_ptr() const;
        std::shared_ptr<DynamicComputeGraph> get_sub_graph_ptr() const;
        virtual TensorList operator()(TensorList& inputs);
        virtual TensorList apply(TensorList& inputs);

    };

    // implementations
    /*
    template<typename ComputeGraphType, typename std::enable_if<std::is_base_of<ComputeGraph, ComputeGraphType>{}, int>::type>
    ComputeNode::ComputeNode(ComputeGraphType& sub_graph)
        : ComputeGraphNode(ComputeGraphNodeType::COMPUTE_GRAPH){
            static_assert(std::is_base_of_v<ComputeGraphType, ComputeGraph>, "ComputeGraphType must derive from ComputeGraph!");
            // call the copy constructor
            _sub_graph_ptr = std::make_shared<ComputeGraphType>(sub_graph);
    }
    */
    /*
    template<typename OP_OR_GRAPH_TYPE>
    ComputeNode::ComputeNode(std::shared_ptr<OP_OR_GRAPH_TYPE> ptr){
        static_assert(std::is_base_of<Operator, OP_OR_GRAPH_TYPE>::value || 
                      std::is_base_of<DynamicComputeGraph, OP_OR_GRAPH_TYPE>::value,
         "OP_OR_GRAPH_ must derive from Operator!");
        if(std::is_base_of<Operator, OP_OR_GRAPH_TYPE>::value){
            _op_ptr.reset(ptr);
            set_node_type(ComputeGraphNodeType::OPERATOR);
        }
        else if(std::is_base_of<DynamicComputeGraph, OP_OR_GRAPH_TYPE>::value){
            _sub_graph_ptr.reset(ptr);
            set_node_type(ComputeGraphNodeType::COMPUTE_GRAPH);
        }
        else{
            exit(EXIT_FAILURE);
            //static_assert(0, "Try to construct compute node without giving op/graph ptr");
        }
    }
    */
}//namespace icdl

#endif
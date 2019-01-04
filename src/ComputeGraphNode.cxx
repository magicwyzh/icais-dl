#include "ComputeGraphNode.h"
#include "ComputeGraph.h"
#include <stdexcept>
namespace icdl{
    ComputeGraphNodeType ComputeGraphNode::get_node_type() const{
        return _ntype;
    }

    TensorList ComputeNode::operator()(TensorList& inputs){
        return apply(inputs);
    }
    
    TensorList ComputeNode::apply(TensorList& inputs){
        if(get_node_type() == ComputeGraphNodeType::OPERATOR){
            return _op_ptr->apply(inputs);
        }
        else if(get_node_type() == ComputeGraphNodeType::COMPUTE_GRAPH){
            return _sub_graph_ptr->apply(inputs);
        }
        else{
            throw std::runtime_error("Unknown node type to execute!");
        }
    }

    std::shared_ptr<Operator> ComputeNode::get_op_ptr() const{
        if(get_node_type() != ComputeGraphNodeType::OPERATOR){
            throw std::runtime_error("Try to get op_ptr from a non-operator node!");
        }
        return _op_ptr;
    }

    std::shared_ptr<DynamicComputeGraph> ComputeNode::get_sub_graph_ptr() const{
        if(get_node_type() != ComputeGraphNodeType::COMPUTE_GRAPH){
            throw std::runtime_error("Try to get op_ptr from a non-operator node!");
        }    
        return _sub_graph_ptr;
    }
    
}//namespace icdl
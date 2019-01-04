#include "ComputeGraph.h"

namespace icdl{

    TensorList ComputeGraph::operator()(TensorList& inputs){
        return apply(inputs);
    }

    void DynamicComputeGraph::add_compute_node(const std::string& name, const ComputeNode& node){
       _compute_nodes.insert(name, node);
    }
    void DynamicComputeGraph::add_compute_node(const std::string& name, const std::shared_ptr<Operator>& op_ptr){
        _compute_nodes.insert(name, ComputeNode(op_ptr));
    }
    void DynamicComputeGraph::add_compute_node(const std::string& name, const std::shared_ptr<DynamicComputeGraph>& sub_graph_ptr){
        _compute_nodes.insert(name, ComputeNode(sub_graph_ptr));
    }

    std::vector<std::pair<std::string, std::shared_ptr<Operator>>> DynamicComputeGraph::get_ops_recursively() const{
        std::vector<std::pair<std::string, ComputeNode>> pairs = _compute_nodes.pairs();
        std::vector<std::pair<std::string, std::shared_ptr<Operator>>> results;
        for(auto& name_node_pair : pairs){
            auto& name = name_node_pair.first;
            auto& node = name_node_pair.second;
            if(node.get_node_type()==ComputeGraphNodeType::COMPUTE_GRAPH){
                // concat
                auto sub_graph_ops = node.get_sub_graph_ptr()->get_ops_recursively(); //recursively
                for(auto& sub_graph_pair: sub_graph_ops){
                    auto demangled_name = name + "->" + sub_graph_pair.first;
                    results.emplace_back(std::make_pair(std::move(demangled_name), std::move(sub_graph_pair.second)));
                }
            }
            else if(node.get_node_type() == ComputeGraphNodeType::OPERATOR){
                // is the std::move right?
                results.emplace_back(std::make_pair(std::move(name), std::move(node.get_op_ptr())));
            }
            else{
                std::cerr << "Find a node in compute graph that is neither op type nor compute_graph type"<<std::endl;
            }
        }
        return results;
    }

    OrderedDict<std::string, ComputeNode>&  DynamicComputeGraph::get_children_nodes(){
        return _compute_nodes;
    }

    ComputeNode& DynamicComputeGraph::get_child_node(const std::string& node_name){
        return _compute_nodes[node_name];
    }

    TensorList SeqDyGraph::apply(TensorList& inputs) {
        auto& x = inputs;
        for(auto node : _compute_nodes.items()){
            x = node.value().apply(x);
        }
        return x;
    }
    SeqDyGraph::SeqDyGraph(const OperatorList& op_list, std::vector<std::string> name_list){
        if(op_list.size() != name_list.size()){
            if(name_list.size() != 0){
                std::cerr << "Try to build a SeqDyGraph with op_list and name_list, but two list has different sizes! "
                    << "Use default name style: unnamed_op<idx>."<<std::endl;
            }
            name_list.clear();
            for(size_t i = 0; i < op_list.size(); i++){
                name_list.emplace_back("unnamed_op" + std::to_string(i));
            }
        }
        for(size_t i = 0; i < op_list.size(); i++){
            add_compute_node(name_list[i], op_list[i]);
        }
    }

    SeqDyGraph::SeqDyGraph(const std::vector<std::pair<std::string, std::shared_ptr<Operator>>>& name_op_pair_list){
        for(auto pair : name_op_pair_list){
            auto name = pair.first;
            auto op_ptr = pair.second;
            add_compute_node(name, op_ptr);
        }
    }
}//namespace icdl
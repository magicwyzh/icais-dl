#include "ComputeGraph.h"
#include <fstream>
#include <iostream>
#include "protos/ComputeGraph.pb.h"
namespace icdl{

    bool DynamicComputeGraph::profile(bool is_profile){
        auto named_ops = get_ops_recursively();
        for(auto named_op : named_ops){
            auto op = named_op.second;
            op->profile(is_profile);
        }
        return is_profile;
    }

    std::vector<std::pair<std::string,ProfileResults>> DynamicComputeGraph::get_profiling_results() const{
        auto named_ops = get_ops_recursively();
        std::vector<std::pair<std::string,ProfileResults>> prof_results;
        for(auto named_op: named_ops){
            prof_results.emplace_back(
                std::make_pair(named_op.first, named_op.second->get_profile_results())
            );
        }
        return prof_results;
    }

    std::string DynamicComputeGraph::demangle_param_name(const std::string& op_name, 
        const std::string& param_name) const{
        return op_name + "." + param_name;
    }
    void DynamicComputeGraph::serialize(const std::string& out_file_name) const{
        GOOGLE_PROTOBUF_VERIFY_VERSION;
        std::fstream output(out_file_name, std::ios::out| std::ios::trunc| std::ios::binary);
        auto graph_proto = serialize();
        graph_proto->SerializeToOstream(&output);
        google::protobuf::ShutdownProtobufLibrary();
    }

    std::shared_ptr<icdl_proto::GraphParams> DynamicComputeGraph::serialize() const{
        auto proto = std::make_shared<icdl_proto::GraphParams>() ;
        auto named_ops = get_ops_recursively();
        for(auto& named_op: named_ops){
            auto& op_name = named_op.first;
            auto& op = named_op.second;
            for(auto named_param: op->get_saved_tensors()){
                auto& param_name = named_param.first;
                auto& tensor = named_param.second;
                const auto demangled_name = demangle_param_name(op_name, param_name);
                (*proto->mutable_graph_params())[demangled_name] = tensor->serialize();
            }
        }
        return proto;
    }
    void DynamicComputeGraph::deserialize(const std::string in_file_name){
        GOOGLE_PROTOBUF_VERIFY_VERSION;

        icdl_proto::GraphParams p;
        std::fstream input(in_file_name, std::ios::in | std::ios::binary);
        p.ParseFromIstream(&input);
        deserialize(p);
        google::protobuf::ShutdownProtobufLibrary();
    }
    void DynamicComputeGraph::deserialize(const icdl_proto::GraphParams& graph_proto){
        auto named_ops = get_ops_recursively();
        std::vector<std::pair<std::string, std::string>> params_not_filled;
        for(auto& named_op: named_ops){
            auto& op_name = named_op.first;
            auto& op = named_op.second;
            for(auto named_param: op->get_saved_tensors()){
                auto& param_name = named_param.first;
                auto& tensor = named_param.second;
                const auto demangled_name = demangle_param_name(op_name, param_name);
                auto& graph_params_map = graph_proto.graph_params();
                // may throw assertion error or key not found error
                try{
                    tensor->deserialize(graph_params_map.at(demangled_name));
                }
                catch(std::exception& e){
                    params_not_filled.emplace_back(std::make_pair(demangled_name, e.what()));
                }
            }
        }
        if(params_not_filled.size()!=0){
            std::cerr << "Caught error when deserializing for params: "<<std::endl;
            for(auto p : params_not_filled){
                std::cerr<< "param: " << p.first
                << ", error msg: " << p.second << std::endl;
            }
            throw(std::runtime_error("Cannot deserialize for a graph."));
        }
    }

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
        auto x = inputs;
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
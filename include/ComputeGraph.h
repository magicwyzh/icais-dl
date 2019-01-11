#ifndef __ICDL_COMPUTE_GRAPH_H__
#define __ICDL_COMPUTE_GRAPH_H__
#include "Tensor.h"
#include "ComputeGraphNode.h"
#include "ordered_dict.h"
#include <string>
#include "Operator.h"
#include <vector>
#include "protos/ComputeGraph.pb.h"

namespace icdl{
    class ComputeGraph{
    public:
        virtual TensorList apply(TensorList& inputs) = 0;
        virtual TensorList operator()(TensorList& inputs);
    };

    class DynamicComputeGraph: public ComputeGraph{
    protected:
        OrderedDict<std::string, ComputeNode> _compute_nodes;
    public:
        virtual void add_compute_node(const std::string& name, const ComputeNode& node);
        virtual void add_compute_node(const std::string& name, const std::shared_ptr<Operator>& op_ptr);
        virtual void add_compute_node(const std::string& name, const std::shared_ptr<DynamicComputeGraph>& sub_graph_ptr);
        bool profile(bool is_profile);
        std::vector<std::pair<std::string,ProfileResults>> get_profiling_results() const;
        // to get ptrs to all operators inside a graph. use this to do something like serilization...
        // each operator should have its name... and the name should shows the nesting relationships in the _compute_nodes
        // name should be like: res1->block1->conv1
        std::vector<std::pair<std::string, std::shared_ptr<Operator>>> get_ops_recursively() const;
        void deserialize(const icdl_proto::GraphParams& graph_proto);
        void deserialize(const std::string in_file_name);
        std::shared_ptr<icdl_proto::GraphParams> serialize() const;
        void serialize(const std::string& out_file_name) const;
        std::string demangle_param_name(const std::string& op_name, const std::string& param_name) const;
        // get the _compute_nodes directly.
        OrderedDict<std::string, ComputeNode>& get_children_nodes();
        ComputeNode& get_child_node(const std::string& node_name);
        virtual ~DynamicComputeGraph(){};
    };

    class SeqDyGraph: public DynamicComputeGraph{
    public:
        virtual TensorList apply(TensorList& inputs) override;
        SeqDyGraph(){};

        // all nodes are ops
        explicit SeqDyGraph(const OperatorList& op_list, std::vector<std::string> name_list = std::vector<std::string>());
        // SeqDyGraph({
        //    {"conv1", icdl::Conv2dOpMake(3,3,3, makeConv2dPytorchImpl())},
        //    {"fc1", icdl::LinearOpMake(16,20, makeLinearPytorchImpl())}
        // });
        explicit SeqDyGraph(const std::vector<std::pair<std::string, std::shared_ptr<Operator>>>& name_op_pair_list);
    };
}
#endif
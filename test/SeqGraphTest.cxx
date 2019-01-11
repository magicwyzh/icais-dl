#include "ComputeGraph.h"
#include <gtest/gtest.h>
#include "torch/torch.h"
#include <memory>
#include "icdl.h"
#include "test_utils.h"
#include <chrono>

class SeqGraphTestCommon : public TestUtils{
public:
static const int64_t batch_size = 2;
static const int64_t in_size = 8;
static const int64_t out_size = 7;
};
const int64_t SeqGraphTestCommon::batch_size;
const int64_t SeqGraphTestCommon::in_size;
const int64_t SeqGraphTestCommon::out_size;
// model:
//                  +---------------------------+  +---------------------------+
// +---------+      |                           |  |                           |     +----------+
// |fc1(8:10)+------>fc2_1(10:20)+-->fc2_2(20:5)+-->fc3_1(5:8)+-->fc3_2(8:10)--+---->+fc4(10:10)|
// +---------+      |                           |  |                           |     +----------+
//                  +---------------------------+  +---------------------------+

class Block1: public icdl::SeqDyGraph{
public: 
    Block1(){
        //test of : virtual void DynamicComputeGraph::add_compute_node(const std::string& name, const std::shared_ptr<Operator> op_ptr);
        add_compute_node("fc2_1", 
            icdl::LinearOpMake(10, 20, icdl::op::makeLinearPytorchImpl())
        );
        add_compute_node("fc2_2",
            icdl::LinearOpMake(20, 5, icdl::op::makeLinearPytorchImpl())
        );

        //linear_params_cpy(_compute_nodes["fc2_1"].get_op_ptr(), pytorch_fc2_1);
        //linear_params_cpy(_compute_nodes["fc2_2"].get_op_ptr(), pytorch_fc2_2);
    }
};

class ICDLModel: public icdl::SeqDyGraph{
public:
    ICDLModel(){
        //test of : virtual void DynamicComputeGraph::add_compute_node(const std::string& name, const ComputeNode&);
        add_compute_node("fc1", icdl::ComputeNode(icdl::LinearOpMake(SeqGraphTestCommon::in_size, 10, icdl::op::makeLinearPytorchImpl())));
        //test of : virtual void DynamicComputeGraph::add_compute_node(const std::string& name, const std::shared_ptr<ComputeGraph> sub_graph_ptr);
        add_compute_node("block1", std::make_shared<Block1>()); //implicitly convert block1 ptr to computegraph ptr
        //test of : SeqDyGraph::SeqDyGraph(std::vector<std::pair<std::string, std::shared_ptr<Operator>>> name_op_pair_list)
        std::shared_ptr<icdl::SeqDyGraph> block2_ptr(new icdl::SeqDyGraph({
            {"fc3_1", icdl::LinearOpMake(5, 8, icdl::op::makeLinearPytorchImpl())},
            {"fc3_2", icdl::LinearOpMake(8, 10, icdl::op::makeLinearPytorchImpl())}
        }));
        
        //test of :  
        //template<typename OP_OR_GRAPH_TYPE>
        //ComputeNode::ComputeNode(std::shared_ptr<OP_OR_GRAPH_TYPE> ptr)
        auto block2_node = icdl::ComputeNode(block2_ptr);
        //test of : virtual void add_compute_node(const std::string& name, const ComputeNode& node);
        add_compute_node("block2", block2_node);
        add_compute_node("fc4", icdl::LinearOpMake(10, SeqGraphTestCommon::out_size, icdl::op::makeLinearPytorchImpl()));
        
        
        //linear_params_cpy(block2_ptr->get_nodes()["fc3_1"].get_op_ptr(), pytorch_fc3_1);
        //linear_params_cpy(block2_ptr->get_nodes()["fc3_2"].get_op_ptr(), pytorch_fc3_2);
        //linear_params_cpy(_compute_nodes["fc4"].get_op_ptr(), pytorch_fc4);
    }
};


class SeqGraphTest: public SeqGraphTestCommon{
protected:
    ICDLModel icdl_model;
    torch::nn::Linear pytorch_fc1 = nullptr;
    torch::nn::Linear pytorch_fc2_1 = nullptr;
    torch::nn::Linear pytorch_fc2_2 = nullptr;
    torch::nn::Linear pytorch_fc3_1 = nullptr;
    torch::nn::Linear pytorch_fc3_2 = nullptr;
    torch::nn::Linear pytorch_fc4 = nullptr;
    torch::Tensor pytorch_input; 
    icdl::TensorList icdl_input;
    void linear_params_cpy(std::shared_ptr<icdl::op::Linear> icdl_linear, torch::nn::Linear& pytorch_linear){
        if(icdl::TensorSize_eq_at_IntList(icdl_linear->get_weight().size(), pytorch_linear->weight.sizes())){
            memcpy(icdl_linear->get_weight().data_ptr(), pytorch_linear->weight.data_ptr(), pytorch_linear->weight.numel()*sizeof(float));
        }
        else{
            std::cerr << "Error when do linear_params_cpy:weight: size not equal!"<<std::endl;
        }
        if(icdl::TensorSize_eq_at_IntList(icdl_linear->get_bias().size(), pytorch_linear->bias.sizes())){
            memcpy(icdl_linear->get_bias().data_ptr(), pytorch_linear->bias.data_ptr(), pytorch_linear->bias.numel()*sizeof(float));
        }
        else{
            std::cerr << "Error when do linear_params_cpy:weight: size not equal!"<<std::endl;
        }
    }

    void linear_params_cpy(std::shared_ptr<icdl::Operator> icdl_linear, torch::nn::Linear& pytorch_linear){
        auto icdl_linear_temp = std::dynamic_pointer_cast<icdl::op::Linear>(icdl_linear);
        if(icdl_linear_temp){
            linear_params_cpy(icdl_linear_temp, pytorch_linear);
        }
        else{
            std::cerr << "Error when do linear_params_cpy: icdl_linear is a nullptr" << std::endl;
        }
    }
    void SetUp() override{
        icdl_model = ICDLModel();
        pytorch_fc1 = torch::nn::Linear(in_size, 10);
        pytorch_fc2_1 = torch::nn::Linear(10, 20);
        pytorch_fc2_2 = torch::nn::Linear(20, 5);
        pytorch_fc3_1 = torch::nn::Linear(5, 8);
        pytorch_fc3_2 = torch::nn::Linear(8, 10);
        pytorch_fc4 = torch::nn::Linear(10, out_size);
        // setup the params
        linear_params_cpy(icdl_model.get_child_node("fc1").get_op_ptr(), pytorch_fc1);
        linear_params_cpy(icdl_model.get_child_node("fc4").get_op_ptr(), pytorch_fc4);
        linear_params_cpy(icdl_model.get_child_node("block1").get_sub_graph_ptr()->get_child_node("fc2_1").get_op_ptr(), pytorch_fc2_1);
        linear_params_cpy(icdl_model.get_child_node("block1").get_sub_graph_ptr()->get_child_node("fc2_2").get_op_ptr(), pytorch_fc2_2);
        linear_params_cpy(icdl_model.get_child_node("block2").get_sub_graph_ptr()->get_child_node("fc3_1").get_op_ptr(), pytorch_fc3_1);
        linear_params_cpy(icdl_model.get_child_node("block2").get_sub_graph_ptr()->get_child_node("fc3_2").get_op_ptr(), pytorch_fc3_2);
        // setupt the input
        pytorch_input = torch::rand({batch_size, in_size}).set_requires_grad(false);
        icdl_input = icdl::TensorList(
            {icdl::Tensor(pytorch_input.data_ptr(), 
                      {static_cast<size_t>(batch_size), static_cast<size_t>(in_size)},   
                      icdl::Float32Descriptor()
                    )
            }
        );
    }
    torch::Tensor pytorch_forward(){
        auto x = pytorch_fc1->forward(pytorch_input);
        x = pytorch_fc2_1->forward(x);
        x = pytorch_fc2_2->forward(x);
        x = pytorch_fc3_1->forward(x);
        x = pytorch_fc3_2->forward(x);
        auto pytorch_output = pytorch_fc4->forward(x);
        return pytorch_output;
    }


};

/*********Test Bodies*******************/


TEST(ComputeNodeTest, InitTest){
    auto node1 = icdl::ComputeNode(icdl::LinearOpMake(SeqGraphTestCommon::in_size, 10, icdl::op::makeLinearPytorchImpl()));
    EXPECT_EQ(node1.get_node_type(), icdl::ComputeGraphNodeType::OPERATOR);
    std::shared_ptr<icdl::SeqDyGraph> block2_ptr(new icdl::SeqDyGraph({
            {"fc3_1", icdl::LinearOpMake(5, 8, icdl::op::makeLinearPytorchImpl())},
            {"fc3_2", icdl::LinearOpMake(8, 10, icdl::op::makeLinearPytorchImpl())}
        }));
    auto node2 = icdl::ComputeNode(block2_ptr);
    EXPECT_EQ(node2.get_node_type(), icdl::ComputeGraphNodeType::COMPUTE_GRAPH);
}

TEST(ComputeGraphTest, AddNodeTest){
    icdl::SeqDyGraph graph;
    graph.add_compute_node("fc2_1", 
        icdl::LinearOpMake(10, 20, icdl::op::makeLinearPytorchImpl())
    );
    graph.add_compute_node("block1", std::make_shared<Block1>());
    const auto nodes_in_graph = graph.get_children_nodes();
    EXPECT_EQ(nodes_in_graph["fc2_1"].get_node_type(), icdl::ComputeGraphNodeType::OPERATOR);
    EXPECT_EQ(nodes_in_graph["block1"].get_node_type(), icdl::ComputeGraphNodeType::COMPUTE_GRAPH);
}

TEST_F(SeqGraphTest, SeDeserilaizeResultsCompareTest){
    auto origin_model_out = icdl_model(icdl_input)[0];

    icdl_model.serialize("test.icdl_model");

    auto new_icdl_model = ICDLModel();
    new_icdl_model.deserialize("test.icdl_model");

    auto new_model_out = new_icdl_model(icdl_input)[0];

    EXPECT_EQ(origin_model_out.nelement(), new_model_out.nelement());
    auto p_old = static_cast<float*>(origin_model_out.data_ptr());
    auto p_new = static_cast<float*>(new_model_out.data_ptr());
    for(size_t i = 0; i < origin_model_out.nelement(); i++){
        EXPECT_FLOAT_EQ(p_old[i], p_new[i]);
    }

}
TEST_F(SeqGraphTest, ForwardTest){
    auto pytorch_output = pytorch_forward();
    auto icdl_output = icdl_model(icdl_input);
    icdl_pytorch_tensor_eq_test(icdl_output[0], pytorch_output);
}

TEST_F(SeqGraphTest, GetNodeTest){
    auto fc1_node = icdl_model.get_child_node("fc1");
    auto block1_node = icdl_model.get_child_node("block1");
    EXPECT_EQ(fc1_node.get_node_type(), icdl::ComputeGraphNodeType::OPERATOR);
    EXPECT_EQ(block1_node.get_node_type(), icdl::ComputeGraphNodeType::COMPUTE_GRAPH);
    auto nodes = icdl_model.get_children_nodes();
    EXPECT_TRUE(nodes.contains("fc1"));
    EXPECT_TRUE(nodes.contains("block1"));
    EXPECT_TRUE(nodes.contains("block2"));
    EXPECT_TRUE(nodes.contains("fc4"));
    EXPECT_EQ(nodes["fc1"].get_node_type(), icdl::ComputeGraphNodeType::OPERATOR);
    EXPECT_EQ(nodes["block1"].get_node_type(), icdl::ComputeGraphNodeType::COMPUTE_GRAPH);
    EXPECT_EQ(nodes["block2"].get_node_type(), icdl::ComputeGraphNodeType::COMPUTE_GRAPH);
    EXPECT_EQ(nodes["fc4"].get_node_type(), icdl::ComputeGraphNodeType::OPERATOR);
}

TEST_F(SeqGraphTest, ProfilerTest){
    icdl_model.profile(false);
    auto profiling_results1 = icdl_model.get_profiling_results();
    for(auto& prof_result1 : profiling_results1){
        auto& result1 = prof_result1.second;
        EXPECT_EQ(result1.get_time_duration_us().count(), 0);
    }

    icdl_model.profile(true);
    auto icdl_output = icdl_model(icdl_input);
    auto profiling_results = icdl_model.get_profiling_results();
    for(auto& prof_result : profiling_results){
        auto& result = prof_result.second;
        EXPECT_GT(result.get_time_duration_us().count(), 0);
    }
}

TEST_F(SeqGraphTest, GetOperatorsTest){
    auto expect_names = std::vector<std::string>{
        "fc1", 
        "block1->fc2_1",
        "block1->fc2_2",
        "block2->fc3_1",
        "block2->fc3_2",
        "fc4"
    };
    auto expect_in_size = std::vector<size_t>{in_size, 10, 20, 5, 8, 10};
    auto expect_out_size = std::vector<size_t>{10, 20, 5, 8, 10, out_size};
    auto icdl_name_op_pairs = icdl_model.get_ops_recursively();
    EXPECT_EQ(expect_names.size(), icdl_name_op_pairs.size());
    for(size_t i = 0; i < expect_names.size(); i++){
        EXPECT_EQ(expect_names[i], icdl_name_op_pairs[i].first);
        auto op = std::dynamic_pointer_cast<icdl::op::Linear>(icdl_name_op_pairs[i].second);
        ASSERT_TRUE(op!=nullptr) << "Get a pointer that cannot be convert to Linear after get_ops_recursively()";
        EXPECT_EQ(op->get_options().in(), expect_in_size[i]);
        EXPECT_EQ(op->get_options().out(), expect_out_size[i]);
    }
}
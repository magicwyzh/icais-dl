#include "ResNet.h"
#include "icdl.h"
#include <math.h>
using namespace icdl::op;
namespace icdl{namespace resnet{

    OpPtr conv3x3(size_t in_planes, size_t out_planes, size_t stride){
        auto options = icdl::op::Conv2dOptions(in_planes, out_planes, 3).stride(stride).padding(1).with_bias(false);
        return icdl::Conv2dOpMake(options);
    }

    OpPtr conv1x1(size_t in_planes, size_t out_planes, size_t stride){
        auto options = icdl::op::Conv2dOptions(in_planes, out_planes, 1).stride(stride).padding(0).with_bias(false);
        return icdl::Conv2dOpMake(options);
    }

    BasicBlock::BasicBlock(size_t inplanes, size_t planes, size_t stride, std::shared_ptr<DynamicComputeGraph> downsample)
    :stride(stride)
    {
        add_compute_node("conv1", conv3x3(inplanes, planes, stride));
        add_compute_node("bn1", icdl::BatchNorm2dOpMake(planes));
        add_compute_node("relu1", icdl::ActivationOpMake(icdl::op::ActivationType::RELU));
        add_compute_node("conv2", conv3x3(planes, planes));
        add_compute_node("bn2", icdl::BatchNorm2dOpMake(planes));
        add_compute_node("relu2", icdl::ActivationOpMake(icdl::op::ActivationType::RELU));
        add_compute_node("eltAdd", icdl::BinaryEltwiseOpOpMake(icdl::op::BinaryEltwiseOpType::ADD));
        if(downsample){
            add_compute_node("downsample", downsample);
        }
    }

    TensorList BasicBlock::apply(TensorList& inputs){
        auto identity = inputs;
        // main path
        auto out = _compute_nodes["conv1"].apply(inputs);
        out = _compute_nodes["bn1"].apply(out);
        out = _compute_nodes["relu1"].apply(out);
        out = _compute_nodes["conv2"].apply(out);
        out = _compute_nodes["bn2"].apply(out);
        // residual path
        auto downsample_ptr = _compute_nodes.find("downsample");
        if(downsample_ptr){
           identity = (*downsample_ptr).operator()(inputs);
        }

        // Combine two TensorLists with move rather than copy
        out.insert(
            out.end(), 
            std::make_move_iterator(identity.begin()), 
            std::make_move_iterator(identity.end())
        );

        // ResAdd
        out = _compute_nodes["eltAdd"].apply(out);
        out = _compute_nodes["relu2"].apply(out);
        return out;
    }

    TensorList BottleNeck::apply(TensorList& inputs){
        auto identity = inputs;
        // main path
        auto out = _compute_nodes["conv1"].apply(inputs);
        out = _compute_nodes["bn1"].apply(out);
        out = _compute_nodes["relu"].apply(out);
        out = _compute_nodes["conv2"].apply(out);
        out = _compute_nodes["bn2"].apply(out);
        out = _compute_nodes["relu"].apply(out);
        out = _compute_nodes["conv3"].apply(out);
        out = _compute_nodes["bn3"].apply(out);
        // residual path
        auto downsample_ptr = _compute_nodes.find("downsample");
        if(downsample_ptr){
           identity = (*downsample_ptr).operator()(inputs);
        }
        // Combine two TensorLists with move rather than copy
        out.insert(
            out.end(), 
            std::make_move_iterator(identity.begin()), 
            std::make_move_iterator(identity.end())
        );

        // ResAdd
        out = _compute_nodes["eltAdd"].apply(out);
        out = _compute_nodes["relu"].apply(out);
        return out;
    }

    BottleNeck::BottleNeck(size_t inplanes, size_t planes, size_t stride, std::shared_ptr<DynamicComputeGraph> downsample)
    :stride(stride)
    {
        add_compute_node("conv1", conv1x1(inplanes, planes));
        add_compute_node("bn1", icdl::BatchNorm2dOpMake(planes));
        add_compute_node("relu", icdl::ActivationOpMake(icdl::op::ActivationType::RELU));
        add_compute_node("conv2", conv3x3(planes, planes,stride));
        add_compute_node("bn2", icdl::BatchNorm2dOpMake(planes));
        add_compute_node("conv3", conv1x1(planes, planes*expansion));
        add_compute_node("bn3", icdl::BatchNorm2dOpMake(planes*expansion));
        add_compute_node("eltAdd", icdl::BinaryEltwiseOpOpMake(icdl::op::BinaryEltwiseOpType::ADD));
        if(downsample){
            add_compute_node("downsample", downsample);
        }
    }

    
    ResNet::ResNet(BlockType block_type, const std::array<size_t, 4>& layers, size_t num_classes){
        inplanes = 64;
        add_compute_node("conv1", icdl::Conv2dOpMake(
            Conv2dOptions(3, 64, 7).stride(2).padding(3).with_bias(false)
        ));
        add_compute_node("bn1", icdl::BatchNorm2dOpMake(64));
        add_compute_node("relu", icdl::ActivationOpMake(ActivationType::RELU));
        add_compute_node("maxpool", icdl::Pooling2dOpMake(
            Pooling2dOptions(PoolType::MAX).kernel_size(3).stride(2).padding(1)
        ));
        /* never change the order of layer1~4 in class declaration !*/
        _make_layer("layer1", block_type, 64, layers[0]);
        _make_layer("layer2",block_type, 128, layers[1]);
        _make_layer("layer3",block_type, 256, layers[2]);
        _make_layer("layer4",block_type, 512, layers[3]);
        add_compute_node("avgpool", icdl::Pooling2dOpMake(
            Pooling2dOptions(PoolType::ADAPTIVE_AVG).output_size({1,1})
        ));
        add_compute_node("fc", icdl::LinearOpMake(
            512 * get_expansion(block_type), num_classes
        ));
    }

    void display_tensor(const icdl::Tensor& tensor){
        auto data_ptr = static_cast<float*>(tensor.data_ptr());
        
        for(size_t i = 0; i < tensor.nelement(); i++){
            //std::cout << data_ptr[i] << " ";
            if(isnan(data_ptr[i])){
                std::cout << "NAN FOUND!" << std::endl;
                return;
            }
        }
        std::cout <<"All are numbers" << std::endl;
        std::cout<<std::endl;
    }
    void display_tensor(const std::string& name, const TensorList& inputs){
        auto t = inputs[0];
        std::cout << name << ":" << std::endl;
        display_tensor(t);
    }

    TensorList ResNet::apply(TensorList& inputs) {
        auto x = _compute_nodes["conv1"].apply(inputs);
        x = _compute_nodes["bn1"](x);
        x = _compute_nodes["relu"](x);
        x = _compute_nodes["maxpool"](x);
        
        x = _compute_nodes["layer1"](x);
        x = _compute_nodes["layer2"](x);
        x = _compute_nodes["layer3"](x);
        x = _compute_nodes["layer4"](x);
        x = _compute_nodes["avgpool"](x);
        auto batch_size = x.at(0).size().at(0);
        ICDL_ASSERT(x.at(0).size().size() == 4, 
            "The output from conv layers in ResNet should be 4D");
        auto flatten_size = (x.at(0).nelement())/(batch_size);
        x[0] = x[0].view({batch_size, flatten_size});
        x = _compute_nodes["fc"](x);
        return x;
    }

    ComputeNode& ResNet::_make_layer(std::string name, BlockType block_type, 
        size_t planes, size_t blocks, size_t stride){
        using name_op_pair_list = std::vector<std::pair<std::string, std::shared_ptr<Operator>>>;

        std::shared_ptr<DynamicComputeGraph> downsample = nullptr;
        if(stride != 1 || this->inplanes != planes * get_expansion(block_type)){
            name_op_pair_list downsample_layers = 
            {
                {"0", conv1x1(this->inplanes, planes*get_expansion(block_type), stride)},
                {"1", BatchNorm2dOpMake(planes * get_expansion(block_type))}
            };
            downsample = std::make_shared<SeqDyGraph>(downsample_layers);
        }
        auto sub_graph = std::make_shared<SeqDyGraph>();
        sub_graph->add_compute_node("0", std::make_shared<BasicBlock>(this->inplanes, planes, stride, downsample));
        this->inplanes = planes*get_expansion(block_type);
        for(size_t i = 1; i < blocks; i++){
            sub_graph->add_compute_node(
                std::to_string(i), //name, 1, 2, 3, ... ,
                std::make_shared<BasicBlock>(this->inplanes, planes)
            );
        }
        return add_compute_node(name, ComputeNode(sub_graph));
    }


}}//namespace icdl::resnet
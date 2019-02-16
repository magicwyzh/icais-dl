#pragma once
#include "icdl.h"
#include <memory>
namespace icdl{namespace resnet{
using OpPtr = std::shared_ptr<icdl::Operator>;

OpPtr conv3x3(size_t in_planes, size_t out_planes, size_t stride = 1);
OpPtr conv1x1(size_t in_planes, size_t out_planes, size_t stride = 1);

class ResNetBlock: public DynamicComputeGraph{
};

class BasicBlock: public ResNetBlock{  
public:
    static const size_t expansion = 1;
    size_t stride;
    //ComputeNode &conv1, &bn1, &relu1, &conv2, &bn2, &relu2, &eltAdd;
    BasicBlock(size_t inplanes, size_t planes, size_t stride = 1, std::shared_ptr<DynamicComputeGraph> downsample = nullptr);
    virtual TensorList apply(TensorList& inputs) override;
};

class BottleNeck: public ResNetBlock{
public:
    static const size_t expansion = 4;
    size_t stride;
    //ComputeNode &conv1, &bn1, &relu, &conv2, &bn2, &conv3, &bn3, &eltAdd;
    BottleNeck(size_t inplanes, size_t planes, size_t stride = 1, std::shared_ptr<DynamicComputeGraph> downsample = nullptr);
    virtual TensorList apply(TensorList& inputs) override;
};

enum class BlockType{
    BasicBlock,
    BottleNeck
};

size_t get_expansion(BlockType block_type){
    return (block_type == BlockType::BasicBlock ? 
        BasicBlock::expansion : BottleNeck::expansion);
}


class ResNet: public DynamicComputeGraph{
private:
    ComputeNode& _make_layer(
        std::string name, BlockType block_type, size_t planes, size_t blocks, size_t stride=1
    );
    size_t inplanes;
    //ComputeNode &conv1, &bn1, &relu, &maxpool;
    //ComputeNode  &bn1, &relu, &maxpool;
    /* never change the order of layers since they should be initialized in this order*/
    //ComputeNode &layer1, &layer2, &layer3, &layer4;
    //ComputeNode &avgpool, &fc;

public:
    virtual TensorList apply(TensorList& inputs) override;
    ResNet(BlockType block_type, const std::array<size_t, 4>& layers, size_t num_classes = 1000);
};


std::shared_ptr<DynamicComputeGraph> resnet18(size_t num_classes){
    static const std::array<size_t, 4> layers= {2,2,2,2};
    auto model = std::make_shared<ResNet>(
        BlockType::BasicBlock, layers, num_classes
    );
    return model;
}
std::shared_ptr<DynamicComputeGraph> resnet50(size_t num_classes){
    static const std::array<size_t, 4> layers= {3,4,6,3};
    auto model = std::make_shared<ResNet>(
        BlockType::BottleNeck, layers, num_classes
    );
    return model;
}

}}
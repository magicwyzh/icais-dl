#include <gtest/gtest.h>
#include "icdl.h"
#include "test_utils.h"
#include "models/ResNet.h"


TEST(ResNetTest, OpNameTest){
    auto model = icdl::resnet::resnet18(1000);
    auto name_tensor_pairs = model->get_all_saved_tensors();
    for(const auto & name : name_tensor_pairs.keys()){
        std::cout << name << std::endl;
    }
}
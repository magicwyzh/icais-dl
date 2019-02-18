#include <gtest/gtest.h>
#include "icdl.h"
#include "test_utils.h"
#include "models/ResNet.h"
#include <random>
#include <cmath>
#include<memory>
using namespace icdl;

struct PBTensorPicker{
    OrderedDict<std::string, std::shared_ptr<icdl::Tensor>> all_tensors;
    void deserialize_tensors(const std::string& file_name);
};

void PBTensorPicker::deserialize_tensors(const std::string& file_name){
    GOOGLE_PROTOBUF_VERIFY_VERSION;

    icdl_proto::GraphParams p;
    std::fstream input(file_name, std::ios::in | std::ios::binary);
    p.ParseFromIstream(&input);
    auto all_tensors_pb = p.graph_params();
    for(auto &name_tensor_pair : all_tensors_pb){
        auto name = name_tensor_pair.first;
        auto tensor_pb = name_tensor_pair.second;
        
        icdl::TensorSize sz;
        for(int i = 0; i < tensor_pb.tensor_size_size();i++){
            sz.emplace_back(tensor_pb.tensor_size(i));
        }
        auto icdl_tensor_ptr = std::make_shared<icdl::Tensor>(sz, Float32Descriptor());
        icdl_tensor_ptr->deserialize(tensor_pb);
        this->all_tensors.insert(name, icdl_tensor_ptr);
    }
    google::protobuf::ShutdownProtobufLibrary();
}

class LayerResultCheckHook: public icdl::OpHook{
public:
    std::string op_name;
    std::shared_ptr<PBTensorPicker> tensor_picker;
    LayerResultCheckHook(const std::string& op_name_, const std::shared_ptr<PBTensorPicker>& picker)
        : op_name(op_name_), tensor_picker(picker){}
    virtual void operator()(Operator *op, TensorList& inputs, TensorList& outputs);
};

void LayerResultCheckHook::operator()(Operator *op, TensorList& inputs, TensorList& outputs){
    //auto correct_tensor = tensor_picker->all_tensors[op_name];
    std::shared_ptr<icdl::Tensor> correct_tensor;
    auto p = tensor_picker->all_tensors.find(op_name);

    if(p != nullptr){
        correct_tensor = *p;
    }
    else{
        std::cout << "Op name: " + op_name << " not found in serialization output.\n"; 
        return;
    }
    auto correct_ptr = static_cast<float*>(correct_tensor->data_ptr());
    auto my_result_ptr = static_cast<float*>(outputs[0].data_ptr());
    ASSERT_TRUE(correct_tensor->size() == outputs[0].size()) << "The output size of Op: " 
    << this->op_name << " not matched! Correct is: \n" << correct_tensor->size() << std::endl
    << "But my result has size:\n" << outputs[0].size() << std::endl;

    for(size_t i = 0; i < correct_tensor->nelement(); i++){
        ASSERT_NEAR(correct_ptr[i], my_result_ptr[i], 1e-4) << "Output of Op:" << this->op_name
        << " not matched! i = " << i << ". Expect: " << correct_ptr[i] << ", but get: " 
        << my_result_ptr[i] <<std::endl;
    }
}

TEST(ResNetTest, Res50LayerResultsTest){
    auto picker = std::make_shared<PBTensorPicker>();
    picker->deserialize_tensors("/mnt/e/projects/icdl/test/test_data/res50_layer_outs.icdl_model");
    auto model = icdl::resnet::resnet50(1000);
    model->deserialize("/mnt/e/projects/icdl/test/test_data/res50_float.icdl_model");
    auto back_util = icdl::PytorchBackEndUtils();
    back_util.set_all_op_backends(*model);
    // add hooks
    for(auto& name_op_pair : model->get_ops_recursively()){
        auto op_name = name_op_pair.first;
        auto op = name_op_pair.second;
        // eltAdd not exist in serialized results, and relu are shared, so the results are not correct.
        if(op_name.find("eltAdd") != std::string::npos || op_name.find("relu") != std::string::npos)
            continue;
        OpHookPtr hook = std::make_shared<LayerResultCheckHook>(op_name, picker);
        op->register_hook(hook);
    }
    // deserialize inputs
    GOOGLE_PROTOBUF_VERIFY_VERSION;
    std::fstream input("/mnt/e/projects/icdl/test/test_data/images.icdl_tensor", std::ios::in | std::ios::binary);
    icdl_proto::Tensor image_pb;
    image_pb.ParseFromIstream(&input);
    auto images = icdl::Tensor({1,3,224,224}, Float32Descriptor());
    images.deserialize(image_pb);
    auto x = TensorList{images};
    // forward and check by the hook
    model->apply(x);
    google::protobuf::ShutdownProtobufLibrary();
}

TEST(ResNetTest, Res18LayerResultsTest){
    auto picker = std::make_shared<PBTensorPicker>();
    picker->deserialize_tensors("/mnt/e/projects/icdl/test/test_data/res18_layer_outs.icdl_model");
    auto model = icdl::resnet::resnet18(1000);
    model->deserialize("/mnt/e/projects/icdl/test/test_data/res18_float.icdl_model");
    auto back_util = icdl::PytorchBackEndUtils();
    back_util.set_all_op_backends(*model);
    // add hooks
    for(auto& name_op_pair : model->get_ops_recursively()){
        auto op_name = name_op_pair.first;
        auto op = name_op_pair.second;
        // eltAdd not exist in serialized results, and relu are shared, so the results are not correct.
        if(op_name.find("eltAdd") != std::string::npos || op_name.find("relu") != std::string::npos)
            continue;
        OpHookPtr hook = std::make_shared<LayerResultCheckHook>(op_name, picker);
        op->register_hook(hook);
    }
    // deserialize inputs
    GOOGLE_PROTOBUF_VERIFY_VERSION;
    std::fstream input("/mnt/e/projects/icdl/test/test_data/images.icdl_tensor", std::ios::in | std::ios::binary);
    icdl_proto::Tensor image_pb;
    image_pb.ParseFromIstream(&input);
    auto images = icdl::Tensor({1,3,224,224}, Float32Descriptor());
    images.deserialize(image_pb);
    auto x = TensorList{images};
    // forward and check by the hook
    model->apply(x);
    google::protobuf::ShutdownProtobufLibrary();
}

TEST(ResNetTest, OpNameTest){
    std::vector<std::string> name_list = {
        "conv1.weight",
        "bn1.weight",
        "bn1.bias",
        "bn1.running_mean",
        "bn1.running_var",
        "layer1->0->conv1.weight",
        "layer1->0->bn1.weight",
        "layer1->0->bn1.bias",
        "layer1->0->bn1.running_mean",
        "layer1->0->bn1.running_var",
        "layer1->0->conv2.weight",
        "layer1->0->bn2.weight",
        "layer1->0->bn2.bias",
        "layer1->0->bn2.running_mean",
        "layer1->0->bn2.running_var",
        "layer1->1->conv1.weight",
        "layer1->1->bn1.weight",
        "layer1->1->bn1.bias",
        "layer1->1->bn1.running_mean",
        "layer1->1->bn1.running_var",
        "layer1->1->conv2.weight",
        "layer1->1->bn2.weight",
        "layer1->1->bn2.bias",
        "layer1->1->bn2.running_mean",
        "layer1->1->bn2.running_var",
        "layer2->0->conv1.weight",
        "layer2->0->bn1.weight",
        "layer2->0->bn1.bias",
        "layer2->0->bn1.running_mean",
        "layer2->0->bn1.running_var",
        "layer2->0->conv2.weight",
        "layer2->0->bn2.weight",
        "layer2->0->bn2.bias",
        "layer2->0->bn2.running_mean",
        "layer2->0->bn2.running_var",
        "layer2->0->downsample->0.weight",
        "layer2->0->downsample->1.weight",
        "layer2->0->downsample->1.bias",
        "layer2->0->downsample->1.running_mean",
        "layer2->0->downsample->1.running_var",
        "layer2->1->conv1.weight",
        "layer2->1->bn1.weight",
        "layer2->1->bn1.bias",
        "layer2->1->bn1.running_mean",
        "layer2->1->bn1.running_var",
        "layer2->1->conv2.weight",
        "layer2->1->bn2.weight",
        "layer2->1->bn2.bias",
        "layer2->1->bn2.running_mean",
        "layer2->1->bn2.running_var",
        "layer3->0->conv1.weight",
        "layer3->0->bn1.weight",
        "layer3->0->bn1.bias",
        "layer3->0->bn1.running_mean",
        "layer3->0->bn1.running_var",
        "layer3->0->conv2.weight",
        "layer3->0->bn2.weight",
        "layer3->0->bn2.bias",
        "layer3->0->bn2.running_mean",
        "layer3->0->bn2.running_var",
        "layer3->0->downsample->0.weight",
        "layer3->0->downsample->1.weight",
        "layer3->0->downsample->1.bias",
        "layer3->0->downsample->1.running_mean",
        "layer3->0->downsample->1.running_var",
        "layer3->1->conv1.weight",
        "layer3->1->bn1.weight",
        "layer3->1->bn1.bias",
        "layer3->1->bn1.running_mean",
        "layer3->1->bn1.running_var",
        "layer3->1->conv2.weight",
        "layer3->1->bn2.weight",
        "layer3->1->bn2.bias",
        "layer3->1->bn2.running_mean",
        "layer3->1->bn2.running_var",
        "layer4->0->conv1.weight",
        "layer4->0->bn1.weight",
        "layer4->0->bn1.bias",
        "layer4->0->bn1.running_mean",
        "layer4->0->bn1.running_var",
        "layer4->0->conv2.weight",
        "layer4->0->bn2.weight",
        "layer4->0->bn2.bias",
        "layer4->0->bn2.running_mean",
        "layer4->0->bn2.running_var",
        "layer4->0->downsample->0.weight",
        "layer4->0->downsample->1.weight",
        "layer4->0->downsample->1.bias",
        "layer4->0->downsample->1.running_mean",
        "layer4->0->downsample->1.running_var",
        "layer4->1->conv1.weight",
        "layer4->1->bn1.weight",
        "layer4->1->bn1.bias",
        "layer4->1->bn1.running_mean",
        "layer4->1->bn1.running_var",
        "layer4->1->conv2.weight",
        "layer4->1->bn2.weight",
        "layer4->1->bn2.bias",
        "layer4->1->bn2.running_mean",
        "layer4->1->bn2.running_var",
        "fc.weight",
        "fc.bias"
    };
    auto model = icdl::resnet::resnet18(1000);
    auto name_tensor_pairs = model->get_all_saved_tensors();
    for(const auto & py_name : name_list){
        //std::cout << name << std::endl;
        auto find = name_tensor_pairs.find(py_name);
        EXPECT_TRUE(find != nullptr) << "Cannot find tensor in ICDL model:" << py_name;
    }
}
/* Since there already LayerOutTest, no need output test.
TEST(ResNetTest, OutputTest){
    GOOGLE_PROTOBUF_VERIFY_VERSION;
    auto model = icdl::resnet::resnet18(1000);
    auto back_util = icdl::PytorchBackEndUtils();
    back_util.set_all_op_backends(*model);
    model->deserialize("/mnt/e/projects/icdl/test/test_data/res18_float.icdl_model");

    std::fstream input("/mnt/e/projects/icdl/test/test_data/images.icdl_tensor", std::ios::in | std::ios::binary);
    icdl_proto::Tensor image_pb, output_pb;
    image_pb.ParseFromIstream(&input);
    std::fstream output_results("/mnt/e/projects/icdl/test/test_data/output.icdl_tensor", std::ios::in | std::ios::binary);
    output_pb.ParseFromIstream(&output_results);

    auto images = icdl::Tensor({1,3,224,224}, Float32Descriptor());
    images.deserialize(image_pb);
    auto output = icdl::Tensor({1,1000}, Float32Descriptor());
    output.deserialize(output_pb);

    auto images_in = TensorList{images};
    auto icdl_output = model->apply(images_in)[0];

    float *icdl_ptr, *pb_ptr;
    icdl_ptr = static_cast<float*>(icdl_output.data_ptr());
    pb_ptr = static_cast<float*>(output.data_ptr());
    bool all_correct = true;

    for(int i = 0; i < 1000; i++){
        //EXPECT_NEAR(icdl_ptr[i], pb_ptr[i], 1e-5);
        if(std::abs(icdl_ptr[i] - pb_ptr[i]) > 1e-4){
            all_correct = false;
            break;
        }
    }
    EXPECT_TRUE(all_correct) << "Not all output are correct for ResNet.";
    google::protobuf::ShutdownProtobufLibrary();
}
*/
/*
TEST(ResNetTest, OutputTest){
    auto model = icdl::resnet::resnet18(10);
    auto back_util = icdl::PytorchBackEndUtils();
    back_util.set_all_op_backends(*model);
    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::normal_distribution<> normal_dist{0,1}; //mean=0, std_dev=2
    auto images = Tensor({1,3,224,224}, Float32Descriptor());
    auto data_ptr_float = static_cast<float*>(images.data_ptr());
    // random init
    for(size_t i = 0; i < images.nelement(); i++){
        data_ptr_float[i] = normal_dist(gen);
    }
    auto dict = model->get_all_saved_tensors();
    for(auto key : model->get_all_saved_tensors().keys()){
        auto param = dict[key];
        data_ptr_float = static_cast<float*>(param->data_ptr());
        if(key.find("running_var") != std::string::npos){
            //found;
            for(size_t i = 0; i < param->nelement();i++){
                data_ptr_float[i] = 1.0;
            }
        }
        else{
            for(size_t i = 0; i < param->nelement();i++){
                data_ptr_float[i] = normal_dist(gen);
            }
        }
    }
    auto x = TensorList{images};
    auto out = model->apply(x);
    auto data_ptr = static_cast<float*>(out[0].data_ptr());
    for(size_t i = 0; i < out[0].nelement();++i){
        std::cout<< data_ptr[i] << std::endl;
    }
}
*/
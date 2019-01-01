
#include <gtest/gtest.h>
#include <random>
#include <memory>
#include "operators/Impls.h"
#include "Operator.h"
#include "torch/torch.h"

TEST(OperatorTest, LinearPyTorchImplTest){
    int64_t batch_size = 8;
    int64_t in_size = 32;
    int64_t out_size = 128;

    auto linear_opt = icdl::op::LinearOptions(in_size, out_size,  icdl::Float32Descriptor());
    icdl::OpImplPtr pytorch_linear_impl = icdl::op::makeLinearPytorchImpl();
    auto icdl_fc_layer = icdl::op::Linear(linear_opt, pytorch_linear_impl);
    
    auto pytorch_fc = torch::nn::Linear(in_size, out_size);
    pytorch_fc->eval();
    auto pytorch_fc_weight = pytorch_fc->weight;
    auto pytorch_fc_bias = pytorch_fc->bias;
    auto weight_ptr = static_cast<float*>(pytorch_fc_weight.data_ptr());

    EXPECT_EQ(pytorch_fc_weight.sizes(), at::IntList({out_size, in_size}));
    EXPECT_EQ(pytorch_fc_bias.sizes(), at::IntList({out_size}));
    auto bias_ptr = static_cast<float*>(pytorch_fc_bias.data_ptr());
    torch::Tensor input = torch::rand({batch_size, in_size});
    input.set_requires_grad(false);
    auto input_ptr = input.data_ptr();
    auto pytorch_correct_output = pytorch_fc->forward(input);
    auto pytorch_out_ptr = static_cast<float*>(pytorch_correct_output.data_ptr());
    // make ICDL tensors
    auto icdl_fc_weight_ptr = icdl_fc_layer.get_weight().data_ptr();
    auto icdl_fc_bias_ptr = icdl_fc_layer.get_bias().data_ptr();
    EXPECT_EQ(pytorch_fc_weight.numel(), in_size*out_size);
    memcpy(icdl_fc_weight_ptr, weight_ptr, pytorch_fc_weight.numel()*sizeof(float));
    memcpy(icdl_fc_bias_ptr, bias_ptr, pytorch_fc_bias.numel()*sizeof(float));
    // from blob
    auto icdl_inputs = icdl::TensorList({icdl::Tensor(input_ptr, {static_cast<size_t>(batch_size), static_cast<size_t>(in_size)}, icdl::Float32Descriptor())});
    auto icdl_outputs = icdl_fc_layer(icdl_inputs);
    auto output = icdl_outputs[0];

    EXPECT_EQ(output.size(), icdl::TensorSize({static_cast<size_t>(batch_size), static_cast<size_t>(out_size)}));
    auto icdl_out_ptr = static_cast<float*>(output.data_ptr());
    for(size_t i = 0; i < output.nelement(); i++){
        EXPECT_EQ(pytorch_out_ptr[i], icdl_out_ptr[i]);
    }

}
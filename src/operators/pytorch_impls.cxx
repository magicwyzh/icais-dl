#ifdef PYTORCH_BACKEND_ENABLE
#include "operators/pytorch_impls.h"
#include "operators/pytorch_backend_utils.h"
#include "icdl_exceptions.h"
#include <vector>
#define TENSOR_PYTORCH_COMPATIBLE_CHECK(tensor, impl_name) do{\
        std::string err_msg = std::string("In ") + std::string(#impl_name) \
            + "::apply," + std::string(#tensor) + std::string("check failed");\
        ICDL_ASSERT(tensor.dtype() == kFloat32, err_msg.c_str());\
        ICDL_ASSERT(tensor.get_data_location() == kCPUMem, err_msg.c_str());\
        ICDL_ASSERT(tensor.get_mem_layout() == kDense, err_msg.c_str());\
    }while(0)

#define OP_SAVED_TENSOR_PYTORCH_COMPATIBLE_CHECK(op_ptr, impl_name)\
    for(auto& pair: op_ptr->get_saved_tensors()){\
        auto pt = pair.second;\
        const auto& tensor_ref = *pt;\
        TENSOR_PYTORCH_COMPATIBLE_CHECK(tensor_ref, impl_name);\
    }

#define CHECK_OUTPUT_SIZE_AND_RETURN(icdl_output, pytorch_output, impl_name)\
    ICDL_ASSERT(TensorSize_eq_at_IntList(icdl_output.size(), pytorch_output.sizes()), \
                "Pytorch output size not matched Operator's in " << #impl_name);\
    memcpy(icdl_output.data_ptr(), pytorch_output.data_ptr(), icdl_output.nelement()*sizeof(float));\
    return {icdl_output}


namespace icdl{namespace op{
TensorList LinearPytorchImpl::apply(Operator* op, TensorList& inputs){
    auto linear_op_ptr = dynamic_cast<Linear*>(op);
    // insanity check
    ICDL_ASSERT(linear_op_ptr!=nullptr, "Dynamic cast of op_ptr failed.");
    ICDL_ASSERT(inputs.size() == 1, "Inputs to LinearPytorchImpl should of size=1");
    TENSOR_PYTORCH_COMPATIBLE_CHECK(linear_op_ptr->get_weight(), LinearPytorchImpl);
    if(linear_op_ptr->get_options().with_bias()){
        TENSOR_PYTORCH_COMPATIBLE_CHECK(linear_op_ptr->get_bias(), LinearPytorchImpl);
        ICDL_ASSERT(linear_op_ptr->get_bias().data_ptr() != nullptr, "Get null bias when there is bias in Linear");
    }
    // convert ICDL Tensor to PyTorch Tensor
    auto weight_torch_tensor = icdl_tensor_to_pytorch_tensor(linear_op_ptr->get_weight());
    auto bias_torch_tensor = icdl_tensor_to_pytorch_tensor(linear_op_ptr->get_bias());
    // actually there should be only one tensor input.
    auto input_torch_tensor = icdl_tensor_to_pytorch_tensor(inputs[0]);
    // core computation.
    auto pytorch_output = torch::linear(input_torch_tensor, weight_torch_tensor, bias_torch_tensor);
    // Allocate ICDL Tensor Memory for copying from PyTorch
    auto icdl_output = icdl::Tensor(linear_op_ptr->output_size(inputs[0].size()),
                                    Float32Descriptor());
    ICDL_ASSERT(TensorSize_eq_at_IntList(icdl_output.size(), pytorch_output.sizes()), "Output size not matched in LinearPytorchImpl");
    // copy from pytorch tensor to icdl tensor, this is because pytorch_output is a local variable
    // the underlying data may be out-of-date when go out of this function.
    memcpy(icdl_output.data_ptr(), pytorch_output.data_ptr(), icdl_output.nelement()*sizeof(float));

    return TensorList({icdl_output});
}

TensorList Conv2dPytorchImpl::apply(Operator* op, TensorList& inputs){
    auto op_ptr = dynamic_cast<Conv2d*>(op);
    ICDL_ASSERT(op_ptr!=nullptr, "Dynamic cast of op_ptr failed.");
    ICDL_ASSERT(inputs.size() == 1, "Inputs to Conv2dPytorchImpl should of size=1");
    TENSOR_PYTORCH_COMPATIBLE_CHECK(op_ptr->get_weight(), Conv2dPytorchImpl);
    if(op_ptr->get_options().with_bias()){
        TENSOR_PYTORCH_COMPATIBLE_CHECK(op_ptr->get_bias(), Conv2dPytorchImpl);
        ICDL_ASSERT(op_ptr->get_bias().data_ptr() != nullptr, "Get null bias when there is bias in Conv2d");
    }
    auto weight_torch_tensor = icdl_tensor_to_pytorch_tensor(op_ptr->get_weight());
    torch::Tensor bias_torch_tensor = {};
    if(op_ptr->get_options().with_bias()){
        bias_torch_tensor = icdl_tensor_to_pytorch_tensor(op_ptr->get_bias());
    }
    auto input_torch_tensor = icdl_tensor_to_pytorch_tensor(inputs[0]);
    auto options = op_ptr->get_options();
    auto pytorch_output = torch::conv2d(input_torch_tensor, weight_torch_tensor, bias_torch_tensor,
        torch::IntList(options.stride()->begin(), options.stride()->size()),
        torch::IntList(options.padding()->begin(), options.stride()->size()),
        torch::IntList(options.dilation()->begin(), options.stride()->size()),
        /*groups*/1
    );
    auto icdl_output = icdl::Tensor(op_ptr->output_size(inputs[0].size()),
                                    Float32Descriptor());
    ICDL_ASSERT(TensorSize_eq_at_IntList(icdl_output.size(), pytorch_output.sizes()), "Output size not matched in Conv2dPytorchImpl");
    memcpy(icdl_output.data_ptr(), pytorch_output.data_ptr(), icdl_output.nelement()*sizeof(float));
    return TensorList{icdl_output};
}

// Activations are all inplace 
TensorList ActivationPytorchImpl::apply(Operator* op, TensorList& inputs){
    auto op_ptr = dynamic_cast<Activation*>(op);
    ICDL_ASSERT(op_ptr!=nullptr, "Dynamic cast of op_ptr failed.");
    ICDL_ASSERT(inputs.size() == 1, "Inputs to ActivationPytorchImpl should be of size = 1");
    TENSOR_PYTORCH_COMPATIBLE_CHECK(inputs[0], ActivationPytorchImpl);
    auto input_torch_tensor = icdl_tensor_to_pytorch_tensor(inputs[0]);
    switch(op_ptr->get_options().act_type()){
        case ActivationType::RELU:
            input_torch_tensor.relu_();
            break;
        case ActivationType::RELU_6:
            input_torch_tensor.clamp_(0, 6);
            break;
        case ActivationType::SIGMOID:
            input_torch_tensor.sigmoid_();
            break;
        case ActivationType::TANH:
            input_torch_tensor.tanh_();
            break;
        default:
            throw std::runtime_error("Invalid Activation Type!");
            break;
    }
    return inputs;
}

TensorList AggregatePytorchImpl::apply(Operator* op, TensorList& inputs){
    ICDL_ASSERT(inputs.size() > 1, "Number of input tensors for Aggregate should > 1");
    auto op_ptr = dynamic_cast<Aggregate*>(op);
    ICDL_ASSERT(op_ptr!=nullptr, "Dynamic cast of Aggregate op_ptr failed.");
    std::vector<torch::Tensor> torch_inputs_list;
    std::vector<TensorSize> size_list;
    for(auto& i : inputs){
        torch_inputs_list.emplace_back(icdl_tensor_to_pytorch_tensor(i));
        size_list.emplace_back(i.size());
    }
    auto is_stack = op_ptr->get_options().stack();
    auto dim = op_ptr->get_options().dim();
    torch::Tensor pytorch_output = torch_inputs_list[0];
    if(!is_stack){
        //cat
        pytorch_output = torch::cat(torch_inputs_list, dim);
    }
    else{
        //stack
        pytorch_output = torch::stack(torch_inputs_list, dim);
    }
    pytorch_output = pytorch_output.contiguous();
    auto icdl_output = icdl::Tensor(op_ptr->output_size(size_list).at(0),
                                    Float32Descriptor());
    ICDL_ASSERT(TensorSize_eq_at_IntList(icdl_output.size(), pytorch_output.sizes()), 
                "Pytorch output size not matched Operator's in AggregatePytorchImpl");
    memcpy(icdl_output.data_ptr(), pytorch_output.data_ptr(), icdl_output.nelement()*sizeof(float));
    return {icdl_output};
}

TensorList BatchNorm2dPytorchImpl::apply(Operator* op, TensorList& inputs){
    auto op_ptr = dynamic_cast<BatchNorm2d*>(op);
    ICDL_ASSERT(op_ptr!=nullptr, "Dynamic cast of BatchNorm2d op_ptr failed.");
    ICDL_ASSERT(inputs.size() == 1, "Inputs to BatchNorm2dPytorchImpl should of size=1");
    OP_SAVED_TENSOR_PYTORCH_COMPATIBLE_CHECK(op_ptr, BatchNorm2dPytorchImpl);

    auto input_torch_tensor = icdl_tensor_to_pytorch_tensor(inputs[0]);
    auto weight_torch_tensor = icdl_tensor_to_pytorch_tensor(op_ptr->get_weight());
    auto bias_torch_tensor = icdl_tensor_to_pytorch_tensor(op_ptr->get_bias());
    auto mean_torch_tensor = icdl_tensor_to_pytorch_tensor(op_ptr->get_running_mean());
    auto var_torch_tensor = icdl_tensor_to_pytorch_tensor(op_ptr->get_running_var());
    auto pytorch_output = torch::batch_norm(input_torch_tensor, weight_torch_tensor,
                      bias_torch_tensor, mean_torch_tensor, var_torch_tensor,
                      false, op_ptr->get_options().momentum(), op_ptr->get_options().eps(),
                      false);
    auto icdl_output = icdl::Tensor(op_ptr->output_size(inputs[0].size()),
                                    Float32Descriptor());

    CHECK_OUTPUT_SIZE_AND_RETURN(icdl_output, pytorch_output, BatchNorm2dPytorchImpl);
}
TensorList BinaryEltwiseOpPytorchImpl::apply(Operator* op, TensorList& inputs){
    auto op_ptr = dynamic_cast<BatchNorm2d*>(op);
    ICDL_ASSERT(op_ptr!=nullptr, "Dynamic cast of BinaryEltwiseOp op_ptr failed.");
    ICDL_ASSERT(inputs.size() == 2, "Inputs to BinaryEltwiseOpImpl should of size=2");

    TENSOR_PYTORCH_COMPATIBLE_CHECK(inputs[0], BinaryEltwiseOpImpl);
    TENSOR_PYTORCH_COMPATIBLE_CHECK(inputs[1], BinaryEltwiseOpImpl);

    auto input0_torch_tensor = icdl_tensor_to_pytorch_tensor(inputs[0]);
    auto input1_torch_tensor = icdl_tensor_to_pytorch_tensor(inputs[1]);
    auto pytorch_output = torch::add(input0_torch_tensor, input1_torch_tensor);
    auto icdl_output = icdl::Tensor(op_ptr->output_size(inputs[0].size()),
                                    Float32Descriptor());
    CHECK_OUTPUT_SIZE_AND_RETURN(icdl_output, pytorch_output, BinaryEltwiseOpPytorchImpl);                          
}

#undef CHECK_OUTPUT_SIZE_AND_RETURN
#undef OP_SAVED_TENSOR_PYTORCH_COMPATIBLE_CHECK
#undef TENSOR_PYTORCH_COMPATIBLE_CHECK
}}//namespace icdl::op
#endif//PYTORCH_BACKEND_ENABLE
#include "operators/LinearImpl.h"
#include "operators/Linear.h"
#ifdef PYTORCH_BACKEND_ENABLE
#include "tensor_utils.h"
#include "torch/torch.h"
#include "operators/pytorch_backend_utils.h"
#endif

namespace icdl{namespace  op{
#ifdef PYTORCH_BACKEND_ENABLE
    TensorList LinearPytorchImpl::apply(Operator* op, TensorList& inputs){
        auto linear_op_ptr = dynamic_cast<Linear*>(op);
        // insanity check
        assert(inputs.size() == 1);
        assert(linear_op_ptr->get_weight().dtype() == kFloat32);
        assert(linear_op_ptr->get_weight().get_data_location() == kCPUMem);
        assert(linear_op_ptr->get_weight().get_mem_layout() == kDense);
        if(linear_op_ptr->get_options().with_bias()){
            assert(linear_op_ptr->get_bias().dtype() == kFloat32);
            assert(linear_op_ptr->get_bias().data_ptr() != nullptr);
        }
        // convert ICDL Tensor to PyTorch Tensor
        auto weight_torch_tensor = icdl_tensor_to_pytorch_tensor(linear_op_ptr->get_weight());
        auto bias_torch_tensor = icdl_tensor_to_pytorch_tensor(linear_op_ptr->get_bias());
        // actually there should be only one tensor input.
        auto input_torch_tensor = icdl_tensor_to_pytorch_tensor(inputs[0]);
        // core computation.
        auto pytorch_output = torch::linear(input_torch_tensor, weight_torch_tensor, bias_torch_tensor);
        // Allocate ICDL Tensor Memory for coping from PyTorch
        auto icdl_output = icdl::Tensor(linear_op_ptr->output_size(inputs[0].size()),
                                        Float32Descriptor());
        assert(TensorSize_eq_at_IntList(icdl_output.size(), pytorch_output.sizes()));
        // copy from pytorch tensor to icdl tensor, this is because pytorch_output is a local variable
        // the underlying data may be out-of-date when go out of this function.
        memcpy(icdl_output.data_ptr(), pytorch_output.data_ptr(), icdl_output.nelement()*sizeof(float));

        return TensorList({icdl_output});
    }
    std::unique_ptr<OperatorImpl> makeLinearPytorchImpl(){
        std::unique_ptr<OperatorImpl> pInv;
        pInv.reset(new LinearPytorchImpl);
        return pInv;
    }
#endif
}} //namespace icdl::op

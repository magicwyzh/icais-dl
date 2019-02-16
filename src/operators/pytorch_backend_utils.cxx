#ifdef PYTORCH_BACKEND_ENABLE
#include "operators/pytorch_backend_utils.h"
#include "operators/operators.h"
#include "operators/pytorch_impls.h"
#include <typeinfo>
#endif
namespace icdl{
#ifdef PYTORCH_BACKEND_ENABLE
    using namespace op;
    bool TensorSize_eq_at_IntList(const TensorSize& icdl_size, const at::IntList& pytorch_size){
        if(icdl_size.size() != pytorch_size.size()){
            return false;
        }
        for(size_t i = 0; i < icdl_size.size(); i++){
            if(static_cast<int64_t>(icdl_size[i]) != pytorch_size[i]){
                return false;
            }
        }
        return true;
    }

    torch::Tensor icdl_tensor_to_pytorch_tensor(const icdl::Tensor& icdl_tensor){
        ASSERT_TENSOR_CONVERT_ICDL_TO_PYTORCH(icdl_tensor);
        return torch::from_blob(icdl_tensor.data_ptr(), ICDL_SIZE_TO_INT_LIST(icdl_tensor.size()));
    }


    void PytorchBackEndUtils::set_op_backend(std::shared_ptr<Operator> op_ptr){

        //auto = std::type_info
        static const auto &conv2d = typeid(Conv2d(1,1,1));
        static const auto &activation = typeid(Activation(ActivationType::RELU));
        static const auto &aggregate = typeid(Aggregate(0));
        static const auto &batchnorm2d = typeid(BatchNorm2d(1));
        static const auto &linear = typeid(Linear(1, 1));
        static const auto &pooling2d = typeid(Pooling2d(PoolType::MAX));
        static const auto &binelt = typeid(BinaryEltwiseOp(BinaryEltwiseOpType::ADD));
        const std::type_info &op_type = typeid(*op_ptr);
        std::shared_ptr<OperatorImpl> impl_ptr;
        if(op_type == conv2d){
            impl_ptr = makeConv2dPytorchImpl();
        }
        else if(op_type == activation){
            impl_ptr = makeActivationPytorchImpl();
        }
        else if(op_type == aggregate){
            impl_ptr = makeAggregatePytorchImpl();
        }
        else if(op_type == batchnorm2d){
            impl_ptr = makeBatchNorm2dPytorchImpl();
        }
        else if(op_type == linear){
            impl_ptr = makeLinearPytorchImpl();
        }
        else if(op_type == pooling2d){
            impl_ptr = makePooling2dPytorchImpl();
        }
        else if(op_type == binelt){
            impl_ptr = makeBinaryEltwiseOpPytorchImpl();
        }
        else{
            throw std::runtime_error(std::string("pytorch backend of: ") + op_type.name() + "has not been added");
        }

        op_ptr->reset_impl(impl_ptr);
    }

    void PytorchBackEndUtils::set_all_op_backends(DynamicComputeGraph &model){
        auto ops = model.get_ops_recursively();
        for(auto& name_op_pair : ops){
            auto op = name_op_pair.second;
            set_op_backend(op);
        }
    }

    void PytorchBackEndUtils::set_all_op_backends(ComputeNode &compute_node){
        if(compute_node.get_node_type() == ComputeGraphNodeType::OPERATOR){
            set_op_backend(compute_node.get_op_ptr());
        }
        else if(compute_node.get_node_type() == ComputeGraphNodeType::COMPUTE_GRAPH){
            set_all_op_backends(*(compute_node.get_sub_graph_ptr()));
        }
        else{
            throw std::runtime_error("Invalid compute node type for pytorch impl");
        }
    }
#endif //PYTORCH_BACKEND_ENABLE
}//namespace icdl
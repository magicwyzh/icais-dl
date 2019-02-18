#pragma once
#include <memory>
#include <vector>
namespace icdl{
    class Operator;
    class Tensor;
    using TensorList = std::vector<Tensor>;
    class OpHook{
    public:
        // pre hook
        virtual void operator()(Operator *op, TensorList& inputs){};
        // post hook
        virtual void operator()(Operator *op, TensorList& inputs, TensorList& outputs){};
    };
    using OpHookPtr = std::shared_ptr<OpHook>;
}//namespace icdl
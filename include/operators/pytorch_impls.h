#pragma once
#ifdef PYTORCH_BACKEND_ENABLE
#include "OperatorImpl.h"
#include "operators/operators.h"
#include "tensor_utils.h"
#include "Operator.h"
#include "pytorch_backend_utils.h"

#define REGISTER_PYTORCH_IMPL(OP_NAME)\
    class OP_NAME##PytorchImpl: public OperatorImpl{\
        virtual TensorList apply(Operator* op, TensorList& inputs) override;\
        virtual std::string name() const override{\
            return std::string(#OP_NAME) + std::string("PytorchImpl");\
        }\
    };\
    OP_IMPL_FACTORY_REGISTER(OP_NAME##PytorchImpl)

namespace icdl{namespace op{
    REGISTER_PYTORCH_IMPL(Activation);
    REGISTER_PYTORCH_IMPL(Aggregate);
    REGISTER_PYTORCH_IMPL(BatchNorm2d);
    REGISTER_PYTORCH_IMPL(BinaryEltwiseOp);
    REGISTER_PYTORCH_IMPL(Conv2d);
    REGISTER_PYTORCH_IMPL(Linear);
    REGISTER_PYTORCH_IMPL(Pooling2d);
}}// namespace icdl::op
#undef REGISTER_PYTORCH_IMPL
#endif
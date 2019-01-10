#ifndef __ICDL_OP_LINEAR_IMPL_H__
#define __ICDL_OP_LINEAR_IMPL_H__
#include "OperatorImpl.h"
#include "Linear.h"
//#define PYTORCH_BACKEND_ENABLE
namespace icdl{ namespace op{
    class LinearImpl: public OperatorImpl{
        virtual std::string name() const override{
            return "LinearImpl";
        }
    };
#ifdef PYTORCH_BACKEND_ENABLE
    class LinearPytorchImpl: public LinearImpl{
        virtual TensorList apply(Operator* op, TensorList& inputs) override;
        virtual std::string name() const override{
            return "LinearPytorchImpl";
        }
    };
    // factory function
    //std::unique_ptr<OperatorImpl> makeLinearPytorchImpl();
    OP_IMPL_FACTORY_REGISTER(LinearPytorchImpl);
#endif
}}
#endif
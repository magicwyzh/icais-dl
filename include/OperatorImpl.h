#ifndef __ICDL_OPERATOR_IMPL_H__
#define __ICDL_OPERATOR_IMPL_H__
#include "Tensor.h"
#include <memory>
#include <exception>

namespace icdl{
#define OP_IMPL_FACTORY_REGISTER(IMPL_NAME) \
    template<typename... Args> \
    std::unique_ptr<OperatorImpl> make##IMPL_NAME(Args&&... args){\
        std::unique_ptr<OperatorImpl> pInv;\
        pInv.reset(new IMPL_NAME(std::forward<Args>(args)...));\
        return pInv;\
    }

    class Operator;
    class OperatorImpl {
    private:
    public:
        // get operator info and params from the pointer op, but need use 
        // dynamic_pointer_cast<XXXOp>(op)
        virtual TensorList apply(Operator* op, TensorList& inputs) = 0;
        virtual std::string name() const{
            return "Unspecified_OperatorImpl";
        }
        virtual ~OperatorImpl() = default;
    };
    
    class EmptyOperatorImpl: public OperatorImpl{
        virtual TensorList apply(Operator* op, TensorList& inputs) override;
        virtual std::string name() const{
            return "EmptyOperatorImpl";
        }
    };

    OP_IMPL_FACTORY_REGISTER(EmptyOperatorImpl);
}//namespace icdl
#endif
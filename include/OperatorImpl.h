#ifndef __ICDL_OPERATOR_IMPL_H__
#define __ICDL_OPERATOR_IMPL_H__
#include "Tensor.h"
#include <memory>
namespace icdl{
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
    
}//namespace icdl
#endif
#ifndef __ICDL_OPERATOR_IMPL_H__
#define __ICDL_OPERATOR_IMPL_H__
#include "Tensor.h"
#include <memory>
#include <exception>
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
    
    class EmptyOperatorImpl: public OperatorImpl{
        virtual TensorList apply(Operator* op, TensorList& inputs) override{
            std::string s = op->type_name() + " Operator is using empty Impl!";
            throw std::runtime_error(s);
        }
        virtual std::string name() const{
            return "EmptyOperatorImpl";
        }
    };
    std::unique_ptr<OperatorImpl> makeEmptyOperatorImpl(){
        std::unique_ptr<OperatorImpl> pInv;
        pInv.reset(new EmptyOperatorImpl);
        return pInv;
    }
}//namespace icdl
#endif
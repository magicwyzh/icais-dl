#include "OperatorImpl.h"
#include "Operator.h"
namespace icdl{
    TensorList EmptyOperatorImpl::apply(Operator* op, TensorList& inputs) {
        std::string s = op->type_name() + " Operator is using empty Impl!";
        throw std::runtime_error(s);
        return {};
    }
    
}//namespace icdl
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
}}//namespace icdl::op
#endif
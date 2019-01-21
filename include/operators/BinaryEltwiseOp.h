#pragma once
#include "arg_utils.h"
#include "Operator.h"

namespace icdl{namespace op{
    enum class BinaryEltwiseOpType: size_t{
        ADD,
        MUL,
        INVALID_OP_TYPE
    };

    struct BinaryEltwiseOpOptions{
        BinaryEltwiseOpOptions(const BinaryEltwiseOpType& op_type): op_type_(op_type){}
        ICDL_ARG(BinaryEltwiseOpType, op_type) = BinaryEltwiseOpType::INVALID_OP_TYPE;
    };

    class BinaryEltwiseOp: public icdl::Operator{
    OP_ADD_OPTIONS(BinaryEltwiseOp);
    OP_ADD_COMMON_FUNCTIONS(BinaryEltwiseOp);
    public:
        BinaryEltwiseOp(const BinaryEltwiseOpOptions& options,
                        OpImplPtr impl = makeEmptyOperatorImpl()):
                        _options(options){}
    };

}
    OP_FACTORY_REGISTER(BinaryEltwiseOp);
}//namespace icdl
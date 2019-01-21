#pragma once
#include "arg_utils.h"
#include "Operator.h"

namespace icdl{namespace op{

    struct AggregateOptions{
        /**
         * @brief Construct a new Aggregate Options object
         * @param dim the aggregate dimension
         * @param stack stack means concat along a new dim
         */
        AggregateOptions(const size_t dim, const bool stack = false): dim_(dim), stack_(stack){}
        ICDL_ARG(size_t, dim);
        ICDL_ARG(bool, stack) = false;
    };

    /**
     * @brief Operator like concat, stack. 
     * 
     */
    class Aggregate: public Operator{
    OP_ADD_OPTIONS(Aggregate);
    OP_ADD_COMMON_FUNCTIONS(Aggregate);
    public:
        Aggregate(const AggregateOptions& options,
                   OpImplPtr impl = makeEmptyOperatorImpl()):
                   _options(options){}
    };
}
    OP_FACTORY_REGISTER(Aggregate);
}//namespace icdl
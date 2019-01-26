#pragma once
#include "Operator.h"
#include "arg_utils.h"

namespace icdl{namespace op{

    enum class PoolType: size_t{
        MAX = 0,
        AVG = 1,
        ADAPTIVE_AVG=2
    };
    static std::array<std::string, 3> pool_type_str{"MaxPooling", "AveragePooling", "AdaptiveAveragePooling"};
    const std::string& PoolType_to_str(const PoolType pool_type);
    struct Pooling2dOptions{
        Pooling2dOptions(const PoolType& pool_type): pool_type_(pool_type){}
        ICDL_ARG(PoolType, pool_type); 
        ICDL_ARG(ExpandingArray<2>, kernel_size) = 2;
        ICDL_ARG(ExpandingArray<2>, stride) = 2;
        ICDL_ARG(ExpandingArray<2>, padding) = 0;
        ICDL_ARG(ExpandingArray<2>, dilation) = 1;
        /**
         * @brief Only used for adaptive average pooling.
         * 
         */
        ICDL_ARG(ExpandingArray<2>, output_size) = {1,1};
    };

    class Pooling2d: public Operator{
    OP_ADD_OPTIONS(Pooling2d);
    public:
        Pooling2d(const Pooling2dOptions& options, OpImplPtr impl = makeEmptyOperatorImpl())
            : Operator(impl), _options(options){}
        virtual TensorSize output_size(const TensorSize& input_size) const;
        virtual std::string type_name() const override{
            return PoolType_to_str(get_options().pool_type());
        }
    };

}
    OP_FACTORY_REGISTER(Pooling2d);
}

#pragma once
#include "arg_utils.h"
#include "Operator.h"
namespace icdl{ namespace op{
    struct Conv2dOptions{
        Conv2dOptions(const size_t input_channels, const size_t output_channels, 
                    ExpandingArray<2> kernel_size) 
            : input_channels_(input_channels), output_channels_(output_channels), 
                kernel_size_(kernel_size) {}

        /// The number of input features (columns of the input matrix).
        ICDL_ARG(size_t, input_channels);
        /// The number of output features to produce (columns of the output matrix).
        ICDL_ARG(size_t, output_channels);
        /// Whether to learn and add a bias after the linear transformation.
        ICDL_ARG(bool, with_bias) = false;

        ICDL_ARG(ExpandingArray<2>, kernel_size);

        ICDL_ARG(ExpandingArray<2>, stride) = 1;

        ICDL_ARG(ExpandingArray<2>, padding) = 0;

        ICDL_ARG(ExpandingArray<2>, dilation) = 1;

        ICDL_ARG(TensorDataDescriptor, param_descriptor) = Float32Descriptor();
    };//Conv2dOptions
    
    class Conv2d: public Operator{
    OP_ADD_TENSOR(weight);
    OP_ADD_TENSOR(bias);
    OP_ADD_OPTIONS(Conv2d);
    OP_ADD_COMMON_FUNCTIONS(Conv2d);
    public:
        Conv2d(const size_t input_channels, const size_t output_channels,
                const ExpandingArray<2>& kernel_size, 
                const ExpandingArray<2>& stride = 1,
                const ExpandingArray<2>& padding = 1,
                const ExpandingArray<2>& dilation = 1,
                const bool with_bias = false,
                OpImplPtr impl = makeEmptyOperatorImpl(), 
                const TensorDataLocation& param_location = kCPUMem,
                const TensorMemLayout& param_mem_layout = kDense
        );
        Conv2d(const Conv2dOptions& options, 
                OpImplPtr impl = makeEmptyOperatorImpl(), 
                const TensorDataLocation& param_location = kCPUMem,
                const TensorMemLayout& param_mem_layout = kDense
        );
        virtual TensorSize output_size(const TensorSize& input_size) const override;
    };

}
    OP_FACTORY_REGISTER(Conv2d); 
}
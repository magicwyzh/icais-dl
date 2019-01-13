#pragma once
#include "Operator.h"
#include "arg_utils.h"
namespace  icdl{ namespace op{

    struct BatchNorm2dOptions {
        /* implicit */ BatchNorm2dOptions(size_t features):features_(features){}
        /// The number of features of the input tensor.
        /// Changing this parameter after construction has no effect.
        ICDL_ARG(size_t, features);
        /// Whether has a learned scale and bias that are applied in an affine
        /// transformation on the input.
        /// Changing this parameter after construction of BatchNorm Operator has no effect.
        ICDL_ARG(bool, affine) = true;
        /// Whether to store and update batch statistics (mean and variance) in the
        /// module. 
        ICDL_ARG(bool, stateful) = true;
        /// The epsilon value added for numerical stability.
        ICDL_ARG(float, eps) = 1e-5;
        /// A momentum multiplier for the mean and variance.
        ICDL_ARG(float, momentum) = 0.1;
        
        ICDL_ARG(TensorDataDescriptor, param_descriptor) = Float32Descriptor();
    };

    class BatchNorm2d: public Operator{
    OP_ADD_TENSOR(weight);
    OP_ADD_TENSOR(bias);
    OP_ADD_TENSOR(running_mean);
    OP_ADD_TENSOR(running_var);
    OP_ADD_OPTIONS(BatchNorm2d);
    OP_ADD_COMMON_FUNCTIONS(BatchNorm2d);
    public:
        BatchNorm2d(const BatchNorm2dOptions& options, 
            OpImplPtr impl = makeEmptyOperatorImpl(), 
            const TensorDataLocation& param_location = kCPUMem,
            const TensorMemLayout& param_mem_layout = kDense
        );
        BatchNorm2d(const size_t features, 
                    bool affine=true, 
                    bool stateful=true, 
                    float eps = 1e-5, 
                    float momentum = 0.1,
                    OpImplPtr impl = makeEmptyOperatorImpl(), 
                    const TensorDataLocation& param_location = kCPUMem,
                    const TensorMemLayout& param_mem_layout = kDense
        );
        virtual TensorSize output_size(const TensorSize& input_size) const override{
            return input_size;
        }
    };
}
    OP_FACTORY_REGISTER(BatchNorm2d); //in icdl namespace 
} //  icdl::op
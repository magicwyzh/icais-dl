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
    ICDL_ARG(double, eps) = 1e-5;
    /// A momentum multiplier for the mean and variance.
    ICDL_ARG(double, momentum) = 0.1;
    };

    class BatchNorm2d{
    OP_ADD_TENSOR(weight);
    OP_ADD_TENSOR(bias);
    OP_ADD_TENSOR(running_mean);
    OP_ADD_TENSOR(running_var);
    OP_ADD_OPTIONS(BatchNorm2d);
    public:
        BatchNorm2d(const BatchNorm2dOptions& options, 
            OpImplPtr impl = makeEmptyOperatorImpl(), 
            const TensorDataLocation& param_location = kCPUMem,
            const TensorMemLayout& param_mem_layout = kDense
        );
        BatchNorm2d(const size_t features,
                    bool affine,
                    bool stateful,
                    bool eps,


        )
    };


}} //  icdl::op
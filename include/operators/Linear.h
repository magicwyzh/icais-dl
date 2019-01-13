#ifndef __ICDL_OP_LINEAR_H__
#define __ICDL_OP_LINEAR_H__
#include "LinearImpl.h"
#include "Operator.h"
#include "arg_utils.h"
#include "tensor_utils.h"
/*
* fc1 = std::make_shared<icdl::op::Linear>(3, 10, makeLinearPytorchImpl());
*/
namespace icdl{ namespace op{
    // properties: in_, out_, with_bias_
    // methods: LinearOptions in(const size_t&), LinearOptions in(size_t&&), const size_t& in(){return in_;}
    struct LinearOptions{
        LinearOptions(size_t in, size_t out) : in_(in), out_(out) {}
        /// The number of input features (columns of the input matrix).
        ICDL_ARG(size_t, in);
        /// The number of output features to produce (columns of the output matrix).
        ICDL_ARG(size_t, out);
        /// Whether to learn and add a bias after the linear transformation.
        ICDL_ARG(bool, with_bias) = true;

        ICDL_ARG(TensorDataDescriptor, param_descriptor) = Float32Descriptor();
    };
    class Linear: public Operator{
    OP_ADD_TENSOR(weight);
    OP_ADD_TENSOR(bias);
    OP_ADD_OPTIONS(Linear);
    OP_ADD_COMMON_FUNCTIONS(Linear);
    public:
        Linear(const size_t in, const size_t out, 
                OpImplPtr linear_impl = makeEmptyOperatorImpl(), 
                const TensorDataDescriptor& param_descriptor=Float32Descriptor(),
                const TensorDataLocation& param_location = kCPUMem,
                const TensorMemLayout& param_mem_layout = kDense);
        Linear(const LinearOptions& options,
                OpImplPtr linear_impl = makeEmptyOperatorImpl(), 
                const TensorDataLocation& param_location = kCPUMem,
                const TensorMemLayout& param_mem_layout = kDense);
        
        virtual TensorSize output_size(const TensorSize& input_size) const override{
            assert(input_size.size() == 2);//one dim for batch, one dim for #neurons.
            return TensorSize({input_size[0], _options.out()});
        }
    };
}
    OP_FACTORY_REGISTER(Linear); //generate a function called "icdl::LinearOpMake->std::shared_ptr<Operator>"
}//namespace icdl::op


#endif
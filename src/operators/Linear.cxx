#include "operators/Linear.h"
namespace icdl{namespace op{
    Linear::Linear(const size_t in, 
                   const size_t out, 
                   OpImplPtr linear_impl, 
                   const TensorDataDescriptor& param_descriptor,
                   const TensorDataLocation& param_location,
                   const TensorMemLayout& param_mem_layout)
        : Linear(LinearOptions(in, out,  param_descriptor), linear_impl){}

    Linear::Linear(const LinearOptions& options, 
                    OpImplPtr linear_impl, 
                    const TensorDataDescriptor& param_descriptor,
                    const TensorDataLocation& param_location,
                    const TensorMemLayout& param_mem_layout)
        : Operator(linear_impl),  _options(options) {
        if(options.with_bias()){
            _bias = Tensor({options.out()}, options.param_descriptor(), param_location, param_mem_layout);
        }
        _weight = Tensor({options.out(), options.in()},  options.param_descriptor(), param_location, param_mem_layout);
    }
}}//namespace icdl::op
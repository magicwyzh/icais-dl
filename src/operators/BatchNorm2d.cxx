#include "operators/BatchNorm2d.h"
namespace icdl{namespace op{

    BatchNorm2d::BatchNorm2d(const BatchNorm2dOptions& options, 
        OpImplPtr impl, 
        const TensorDataLocation& param_location,
        const TensorMemLayout& param_mem_layout
    ): Operator(impl), _options(options){
        if(options.affine()){
            _weight = Tensor({options.features()}, options.param_descriptor(), 
                            param_location, param_mem_layout);
            _register_tensor("weight", &_weight);
            _bias = Tensor({options.features()}, options.param_descriptor(), 
                            param_location, param_mem_layout);
            _register_tensor("bias", &_bias);
        }
        _running_mean = Tensor({options.features()}, options.param_descriptor(), 
                            param_location, param_mem_layout);
        _running_var = Tensor({options.features()}, options.param_descriptor(), 
                            param_location, param_mem_layout);
        _register_tensor("running_mean", &_running_mean);
        _register_tensor("running_var", &_running_var);
    }

    BatchNorm2d::BatchNorm2d(const size_t features, 
                bool affine, 
                bool stateful, 
                float eps, 
                float momentum,
                OpImplPtr impl, 
                const TensorDataLocation& param_location,
                const TensorMemLayout& param_mem_layout
    ):BatchNorm2d(BatchNorm2dOptions(features).affine(affine).stateful(stateful).eps(eps).momentum(momentum),
                   impl, param_location, param_mem_layout){}

}}//namespace icdl::op
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
        LinearOptions(size_t in, size_t out, TensorDataDescriptor param_descriptor) 
            : in_(in), out_(out), param_descriptor_(param_descriptor) {}
        /// The number of input features (columns of the input matrix).
        ICDL_ARG(size_t, in);
        /// The number of output features to produce (columns of the output matrix).
        ICDL_ARG(size_t, out);
        /// Whether to learn and add a bias after the linear transformation.
        ICDL_ARG(bool, with_bias) = true;

        ICDL_ARG(TensorDataDescriptor, param_descriptor);
    };
    class Linear: public Operator{
    private:
        LinearOptions _options;
        Tensor _weight;
        Tensor _bias;
    public:
        Linear(const size_t in, const size_t out, 
                OpImplPtr linear_impl, 
                const TensorDataDescriptor& param_descriptor=Float32Descriptor(),
                const TensorDataLocation& param_location = kCPUMem,
                const TensorMemLayout& param_mem_layout = kDense);
        Linear(const LinearOptions& options,
                OpImplPtr linear_impl, 
                const TensorDataDescriptor& param_descriptor=Float32Descriptor(),
                const TensorDataLocation& param_location = kCPUMem,
                const TensorMemLayout& param_mem_layout = kDense);
        
        virtual std::string type_name() const override{
            return "Linear";
        }
        virtual std::vector<TensorSize> output_size(const std::vector<TensorSize>& input_sizes) const{
            std::vector<TensorSize> sizes;
            for(auto i_size : input_sizes){
                sizes.emplace_back(output_size(i_size));
            }
            return sizes;
        }
        virtual TensorSize output_size(const TensorSize& input_size) const{
            assert(input_size.size() == 2);//one dim for batch, one dim for #neurons.
            return TensorSize({input_size[0], _options.out()});
        }
        const LinearOptions& get_options() const{
            return _options;
        }
        const Tensor& get_weight() const{
            return _weight;
        }
        const Tensor& get_bias() const{
            return _bias;
        }
    };

    
}
    OP_FACTORY_REGISTER(Linear); //generate a function called "icdl::LinearOpMake->std::shared_ptr<Operator>"
}//namespace icdl::op


#endif
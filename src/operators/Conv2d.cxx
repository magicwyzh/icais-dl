#include "operators/Conv2d.h"
#include "Operator.h"
namespace icdl{namespace op{
    TensorSize Conv2d::output_size(const TensorSize& input_size) const{
        // lambda
        auto out_size1d_options = [](const TensorSize& in_size, const Conv2dOptions& options, const int dim){
            auto out_size1d = [](size_t Hin, int64_t padding, int64_t dilation, int64_t kernel_size, int64_t stride){
                return ((Hin+2*padding-dilation*(kernel_size-1)-1)/stride) + 1;
            };
            return out_size1d(
                in_size[dim],
                options.padding_->at(dim - 2), // first two dims are batch, in channels
                options.dilation_->at(dim - 2),
                options.kernel_size_->at(dim - 2),
                options.stride_->at(dim - 2)
            );
        };

        auto out_size = TensorSize{input_size[0], 
                get_options().output_channels(),
                out_size1d_options(input_size, _options, 2), 
                out_size1d_options(input_size, _options, 3)};
        return out_size;
    }

    Conv2d::Conv2d(const Conv2dOptions& options, 
            OpImplPtr impl, 
            const TensorDataLocation& param_location,
            const TensorMemLayout& param_mem_layout
    ): Operator(impl), _options(options){
        if(options.with_bias()){
            _bias = Tensor({options.output_channels()}, options.param_descriptor(), param_location, param_mem_layout);
            _register_tensor("bias", &_bias);
        }
        auto ksize = options.kernel_size();
        TensorSize weight_size{options.output_channels(), options.input_channels(), static_cast<size_t>(ksize->at(0)), static_cast<size_t>(ksize->at(1))};
        _weight = Tensor(weight_size, options.param_descriptor(), param_location, param_mem_layout);
        _register_tensor("weight", &_weight);
    }

    Conv2d::Conv2d( const size_t input_channels, 
            const size_t output_channels,
            const ExpandingArray<2>& kernel_size, 
            const ExpandingArray<2>& stride,
            const ExpandingArray<2>& padding,
            const ExpandingArray<2>& dilation,
            const bool with_bias,
            OpImplPtr impl, 
            const TensorDataLocation& param_location,
            const TensorMemLayout& param_mem_layout
    ): Conv2d(Conv2dOptions(input_channels, output_channels, kernel_size).stride(stride).padding(padding).dilation(dilation).with_bias(with_bias), 
        impl, param_location, param_mem_layout){}
    
}}//icdl::op
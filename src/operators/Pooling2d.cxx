#include "operators/Pooling2d.h"

namespace icdl{namespace op{

const std::string& PoolType_to_str(const PoolType pool_type){
    return pool_type_str.at(static_cast<size_t>(pool_type));
}

TensorSize Pooling2d::output_size(const TensorSize& input_size) const{
        // lambda
        auto out_size1d_options = [](const TensorSize& in_size, const Pooling2dOptions& options, const int dim){
            auto out_size1d = [](size_t Hin, int64_t padding, int64_t dilation, int64_t kernel_size, int64_t stride){
                return ((Hin+2*padding-dilation*(kernel_size-1)-1)/stride) + 1;
            };
            return out_size1d(
                in_size[dim],
                options.padding_->at(dim),
                options.dilation_->at(dim),
                options.kernel_size_->at(dim),
                options.stride_->at(dim)
            );
        };
        return {out_size1d_options(input_size, _options, 0), out_size1d_options(input_size, _options, 1)};
}

}}
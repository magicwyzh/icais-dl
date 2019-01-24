#include "operators/Aggregate.h"

namespace icdl{namespace op{
std::vector<TensorSize> Aggregate::output_size(const std::vector<TensorSize>& input_sizes) const{
    auto stack = _options.stack();
    auto dim = _options.dim();
    auto num_dim = input_sizes[0].size();
    auto dim_sum = 0;
    auto reference_size = input_sizes[0];
    for(auto &s : input_sizes){
        ICDL_ASSERT(num_dim == s.size(), 
            "TensorSizes for Aggregate out_size should have the same number of dims of "
            << num_dim << "But meet one of the tensor has num_dim = " << s.size());
        for(size_t i = 0; i < num_dim; i++){
            if(stack || i != dim){
                ICDL_ASSERT(s.at(i) == reference_size.at(i), 
                    "More than one dims are not the same for aggregation out_size compute"
                );
            }
        }
        // sum dim if required
        dim_sum += s.at(dim);
    }
    if(!stack){
        // concat
        reference_size[dim]  = dim_sum;
    }
    else{
        //stack
        reference_size.insert(reference_size.begin()+ dim, input_sizes.size());
    }
    return {reference_size};
}

TensorSize Aggregate::output_size(const TensorSize& input_size) const{
    if(!_options.stack()){
        return input_size;
    }
    else{
        auto new_size = input_size;
        new_size.insert(new_size.begin() + _options.dim(), 1);
        return new_size;
    }
}
}}//namespace icdl::op
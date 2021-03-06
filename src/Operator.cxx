#include "Operator.h"

namespace icdl{
    int64_t Operator::compute_complexity(const TensorSize& input_size) const{
        return 0;
    }

    void Operator::_register_tensor(const std::string& tensor_name, Tensor* tensor_ptr){
        _saved_tensors.emplace_back(std::make_pair(tensor_name, tensor_ptr));
    }

    std::list<OpHookPtr>::iterator Operator::register_hook(OpHookPtr& hook){
        return _hooks.insert(_hooks.end(), hook);
    }

    void Operator::remove_all_hooks(){
        _hooks.clear();
    }

    std::string Operator::type_name() const{
            return typeid(*this).name();
    }
    
    void Operator::reset_impl(OpImplPtr impl_ptr){
        _impl = impl_ptr;
    }

    TensorList Operator::operator()(TensorList & inputs){
        return apply(inputs);
    }

    TensorList Operator::apply(TensorList& inputs){
        // RAII profiling
        Profiler prof(this);
        for(auto& pre_hook : _hooks){
            (*pre_hook)(this, inputs);
        }
        
        auto outputs = _impl->apply(this, inputs);

        for(auto& post_hook : _hooks){
            (*post_hook)(this, inputs, outputs);
        }
        return outputs;
    }

    bool Operator::profile(bool is_profile){
        _profile = is_profile;
        return _profile;
    }

    std::vector<std::pair<std::string, Tensor*>> Operator::get_saved_tensors(){
        return _saved_tensors;
    }

    ProfileResults Operator::get_profile_results(){
        return _prof_results;
    }

    std::vector<TensorSize> Operator::output_size(const std::vector<TensorSize>& input_sizes) const{
        std::vector<TensorSize> sizes;
        for(auto& i_size : input_sizes){
            sizes.emplace_back(output_size(i_size));
        }
        return sizes;
    }
}// namespace icdl
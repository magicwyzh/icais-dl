#ifndef __ICDL_OPERATOR_H__
#define __ICDL_OPERATOR_H__
#include "OperatorImpl.h"
namespace icdl{
    class Operator;
    using OpImplPtr = std::shared_ptr<OperatorImpl>;
    using OperatorList = std::vector<std::shared_ptr<Operator>>;
    class Operator{
    protected:
        OpImplPtr _impl;
        std::vector<std::pair<std::string, Tensor*>> _saved_tensors;
        void _register_tensor(const std::string& tensor_name, Tensor* tensor_ptr){
            _saved_tensors.emplace_back(std::make_pair(tensor_name, tensor_ptr));
        }
    public:
        TensorList operator()(TensorList & inputs){
            return apply(inputs);
        }
        TensorList apply(TensorList& inputs){
            return _impl->apply(this, inputs);
        }
        // would be better to override this function to give a pretty name.
        virtual std::string type_name() const{
            return typeid(*this).name();
        }
        void reset_impl(OpImplPtr impl_ptr){
            _impl = impl_ptr;
        }
        Operator(const OpImplPtr& impl = makeEmptyOperatorImpl()): _impl(impl){}
        virtual std::vector<TensorSize> output_size(const std::vector<TensorSize>& input_sizes) const = 0;
        virtual TensorSize output_size(const TensorSize& input_size) const = 0;
        virtual ~Operator() = default;

        std::vector<std::pair<std::string, Tensor*>> get_saved_tensors(){
            return _saved_tensors;
        }
    };
}//namespace icdl
#endif
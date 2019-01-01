#ifndef __ICDL_OPERATOR_H__
#define __ICDL_OPERATOR_H__
#include "OperatorImpl.h"
namespace icdl{
    using OpImplPtr = std::shared_ptr<OperatorImpl>;
    class Operator{
    protected:
        OpImplPtr _impl;
    public:
        TensorList operator()(TensorList & inputs){
            return _impl->apply(this, inputs);
        }
        // would be better to override this function to give a pretty name.
        virtual std::string name() const{
            return typeid(*this).name();
        }
        void reset_impl(OpImplPtr impl_ptr){
            _impl = impl_ptr;
        }
        Operator(const OpImplPtr& impl): _impl(impl){}
        virtual std::vector<TensorSize> output_size(const std::vector<TensorSize>& input_sizes) const = 0;
        virtual TensorSize output_size(const TensorSize& input_size) const = 0;
        virtual ~Operator() = default;
    };
}//namespace icdl
#endif
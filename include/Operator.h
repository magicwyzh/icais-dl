#ifndef __ICDL_OPERATOR_H__
#define __ICDL_OPERATOR_H__
#include "OperatorImpl.h"
#include "Profiler.h"
namespace icdl{
    class Operator;
    class Profiler;
    using OpImplPtr = std::shared_ptr<OperatorImpl>;
    using OperatorList = std::vector<std::shared_ptr<Operator>>;
    class Operator{
        //to let the impl able to get more info about operator
        friend class OperatorImpl;
        friend class Profiler;
    protected:
        OpImplPtr _impl;
        std::vector<std::pair<std::string, Tensor*>> _saved_tensors;
        ProfileResults _prof_results;
        bool _profile{false};
        void _register_tensor(const std::string& tensor_name, Tensor* tensor_ptr);
    public:
        Operator(const OpImplPtr& impl = makeEmptyOperatorImpl()): _impl(impl){}
        TensorList operator()(TensorList & inputs);
        TensorList apply(TensorList& inputs);
        // would be better to override this function to give a pretty name.
        virtual std::string type_name() const;
        void reset_impl(OpImplPtr impl_ptr);
        virtual std::vector<TensorSize> output_size(const std::vector<TensorSize>& input_sizes) const = 0;
        virtual TensorSize output_size(const TensorSize& input_size) const = 0;
        virtual int64_t compute_complexity(const TensorSize& input_size) const;
        virtual ~Operator() = default;
        bool profile(bool is_profile);
        std::vector<std::pair<std::string, Tensor*>> get_saved_tensors();
        ProfileResults get_profile_results();
    };

#define OP_FACTORY_REGISTER(OPNAME) \
    template <typename... Args> \
    std::shared_ptr<icdl::Operator> OPNAME##OpMake(Args&&... args) { \
        return std::make_shared<icdl::op::OPNAME>(std::forward<Args>(args)...); \
    }

}//namespace icdl
#endif
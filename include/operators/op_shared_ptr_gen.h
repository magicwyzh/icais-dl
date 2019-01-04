#ifndef __ICDL_OP_SHARED_PTR_GEN_H__
#define __ICDL_OP_SHARED_PTR_GEN_H__

#define OP_SHARED_PTR_MAKE(OPNAME) \
    template <typename... Args> \
    std::shared_ptr<icdl::Operator> OPNAME##OpMake(Args&&... args) { \
        return std::make_shared<icdl::op::OPNAME>(std::forward<Args>(args)...); \
    }

#endif
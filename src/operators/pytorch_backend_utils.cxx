#include "operators/pytorch_backend_utils.h"
namespace icdl{
#ifdef PYTORCH_BACKEND_ENABLE
    bool TensorSize_eq_at_IntList(const TensorSize& icdl_size, const at::IntList& pytorch_size){
        if(icdl_size.size() != pytorch_size.size()){
            return false;
        }
        for(size_t i = 0; i < icdl_size.size(); i++){
            if(static_cast<int64_t>(icdl_size[i]) != pytorch_size[i]){
                return false;
            }
        }
        return true;
    }

    torch::Tensor icdl_tensor_to_pytorch_tensor(const icdl::Tensor& icdl_tensor){
        ASSERT_TENSOR_CONVERT_ICDL_TO_PYTORCH(icdl_tensor);
        return torch::from_blob(icdl_tensor.data_ptr(), ICDL_SIZE_TO_INT_LIST(icdl_tensor.size()));
    }
#endif //PYTORCH_BACKEND_ENABLE
}//namespace icdl
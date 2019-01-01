#ifndef __ICDL_PYTORCH_BACKEND_UTILS_H__
#define __ICDL_PYTORCH_BACKEND_UTILS_H__
#ifdef PYTORCH_BACKEND_ENABLE
#include "torch/torch.h"
#include "tensor_utils.h"
#include "Tensor.h"
#endif
namespace icdl{
#ifdef PYTORCH_BACKEND_ENABLE
    // generate a temp int_list to create a torch::Tensor from blob
    #define ICDL_SIZE_TO_INT_LIST(icdl_size) \
            at::IntList(reinterpret_cast<int64_t*>(icdl_size.data()), icdl_size.size())

    // check if an ICDL tensor can safely be converted to PyTorch Tensor
    #define ASSERT_TENSOR_CONVERT_ICDL_TO_PYTORCH(icdl_tensor) \
            assert(icdl_tensor.dtype() == kFloat32); \
            assert(icdl_tensor.get_data_location() == kCPUMem); \
            assert(icdl_tensor.get_mem_layout() == kDense)
    // compare a ICDL TensorSize to at::IntList. The latter is used to save tensor size in Pytorch
    bool TensorSize_eq_at_IntList(const TensorSize& icdl_size, const at::IntList& pytorch_size);
    // generate a pytorch tensor from blob.
    torch::Tensor icdl_tensor_to_pytorch_tensor(const icdl::Tensor& icdl_tensor);
#endif
}
#endif
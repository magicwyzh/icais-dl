#include "test_utils.h"
#include "Tensor.h"
#include "torch/torch.h"
#include "operators/pytorch_backend_utils.h"

bool TestUtils::icdl_pytorch_tensor_dtype_same(const icdl::Tensor& icdl_tensor, const torch::Tensor& pytorch_tensor){
    if(pytorch_tensor.dtype() == torch::kFloat32){
        return icdl_tensor.dtype() == icdl::kFloat32;
    }
    else if(pytorch_tensor.dtype() == torch::kHalf){
        return icdl_tensor.dtype() == icdl::kFloat16;
    }
    else{
        return false;
    }
}
void TestUtils::icdl_pytorch_tensor_eq_test(const icdl::Tensor& icdl_tensor, const torch::Tensor& pytorch_tensor){
    ASSERT_TRUE(
        icdl::TensorSize_eq_at_IntList(icdl_tensor.size(), pytorch_tensor.sizes())
    ) << "Tensor Size not equal";
    ASSERT_TRUE(icdl_pytorch_tensor_dtype_same(icdl_tensor, pytorch_tensor));
    ASSERT_EQ(icdl_tensor.dtype(), icdl::kFloat32);
    auto icdl_ptr = static_cast<float*>(icdl_tensor.data_ptr());
    auto pytorch_ptr = static_cast<float*>(pytorch_tensor.data_ptr());
    for(size_t i = 0; i < icdl_tensor.nelement(); i++){
        EXPECT_FLOAT_EQ(pytorch_ptr[i], icdl_ptr[i]) << "Expecte Float EQ failed at index "<<i;
    }
}
void TestUtils::icdl_pytorch_tensor_near_test(const icdl::Tensor& icdl_tensor, const torch::Tensor& pytorch_tensor){
    ASSERT_TRUE(
        icdl::TensorSize_eq_at_IntList(icdl_tensor.size(), pytorch_tensor.sizes())
    ) << "Tensor Size not equal";
    ASSERT_TRUE(icdl_pytorch_tensor_dtype_same(icdl_tensor, pytorch_tensor));
    ASSERT_EQ(icdl_tensor.dtype(), icdl::kFloat32);
    auto icdl_ptr = static_cast<float*>(icdl_tensor.data_ptr());
    auto pytorch_ptr = static_cast<float*>(pytorch_tensor.data_ptr());
    for(size_t i = 0; i < icdl_tensor.nelement(); i++){
        EXPECT_NEAR(pytorch_ptr[i], icdl_ptr[i], 1e-6);
    }
}

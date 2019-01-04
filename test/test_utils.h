#pragma once 
#include <gtest/gtest.h>
#include "icdl.h"
class TestUtils: public ::testing::Test{
protected:
    bool icdl_pytorch_tensor_dtype_same(const icdl::Tensor& icdl_tensor, const torch::Tensor& pytorch_tensor);
    void icdl_pytorch_tensor_eq_test(const icdl::Tensor& icdl_tensor, const torch::Tensor& pytorch_tensor);
    void icdl_pytorch_tensor_near_test(const icdl::Tensor& icdl_tensor, const torch::Tensor& pytorch_tensor);
};



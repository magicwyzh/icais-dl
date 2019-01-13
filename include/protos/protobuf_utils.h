#pragma once
#include "tensor_utils.h"
#include "Tensor.pb.h"
namespace icdl{
    TensorMemLayout proto_mem_layout_to_icdl_layout(const icdl_proto::Tensor_TensorMemLayout proto_layout);
    TensorDataType proto_dtype_to_icdl_dtype(const icdl_proto::TensorDataDescriptor_TensorDataType proto_dtype);
}
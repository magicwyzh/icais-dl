#include "protos/protobuf_utils.h"
namespace icdl{
    TensorDataType proto_dtype_to_icdl_dtype(const icdl_proto::TensorDataDescriptor_TensorDataType proto_dtype){
        TensorDataType proto_data_type;
        switch(proto_dtype){
            case icdl_proto::TensorDataDescriptor_TensorDataType_FLOAT_32: proto_data_type = kFloat32; break;
            case icdl_proto::TensorDataDescriptor_TensorDataType_FLOAT_16: proto_data_type = kFloat16; break;
            case icdl_proto::TensorDataDescriptor_TensorDataType_FIXPOINT: proto_data_type = kFixpoint; break;
            case icdl_proto::TensorDataDescriptor_TensorDataType_INVALID_DTYPE: proto_data_type = TensorDataType::INVALID_DTYPE;break;
            default: {
                throw std::runtime_error("The protobuf's TensorDataType is not compatible with the definition in icdl source.");
            }
        }
        return proto_data_type;
    }

    TensorMemLayout proto_mem_layout_to_icdl_layout(const icdl_proto::Tensor_TensorMemLayout proto_layout){
        TensorMemLayout icdl_layout;
        switch(proto_layout){
            case icdl_proto::Tensor_TensorMemLayout_DENSE_LAYOUT: icdl_layout = kDense; break;
            case icdl_proto::Tensor_TensorMemLayout_INVALID_LAYOUT: icdl_layout = TensorMemLayout::INVALID_LAYOUT;break;
            case icdl_proto::Tensor_TensorMemLayout_SPARSE_LAYOUT: icdl_layout = kSparse; break;
            default: {
                throw(std::runtime_error("The protobuf's TensorMemLayout is not compatible with the definition in icdl source."));
            }
        }
        return icdl_layout;
    }
}
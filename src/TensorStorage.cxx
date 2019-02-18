#include "TensorStorage.h"
#include <string.h>
#include "tensor_utils.h"
#include "accelerator_memory.h"
#include "protos/protobuf_utils.h"
#include "icdl_exceptions.h"
namespace icdl{
    static bool is_little_endian(){
        int n = 1;
        return *(char*)&n == 1;
    }

    void TensorStorage::deserialize(const icdl_proto::TensorStorage& storage_proto){
        ICDL_ASSERT(is_little_endian(), "Currently only support little endian machine (as x86) when deserializing!");
        ICDL_ASSERT(data_location_ == kCPUMem, 
            "A TensorStorage for deserialization must be in CPUMem, but meets a " 
            << enum_to_string(data_location_));
        
        TensorDataType dtype_pb = proto_dtype_to_icdl_dtype(storage_proto.data_descriptor().dtype());
        ICDL_ASSERT(get_data_type() == dtype_pb, "DataType Not match when deserializing TensorStorage: " 
            << "Protobuf dtype=" << enum_to_string(dtype_pb) << ", icdl storage dtype = " << enum_to_string(get_data_type())
        );
        TensorDataDescriptor descrpt_pb;
        // set descriptor
        auto proto_descriptor = icdl_proto::TensorDataDescriptor();
        
        // sanity check
        if(get_data_type() == kFloat32 || get_data_type() == kFloat16){
            ICDL_ASSERT(storage_proto.data_descriptor().has_flo_point(), 
                "A protobuf object for FloatTensorStorage Must has float point data represent");
                
            auto r = storage_proto.data_descriptor().flo_point();
            auto my_flo = get_data_descriptor().get_represent().flo_point;
            ICDL_ASSERT(get_data_descriptor().get_represent().flo_point == 
                FloatpointRepresent(r.total_bits(), r.is_signed(), r.exp_bits(), r.mantissa_bits()),
                "FloatStorage has different data represent with that in protobuf object when deserializing"
                << r.total_bits() << std::endl << r.is_signed() << std::endl << r.exp_bits() << std::endl << r.mantissa_bits() << std::endl << my_flo.total_bits << std::endl<<my_flo.is_signed << std::endl
                << my_flo.exp_bits << std::endl << my_flo.mantissa_bits << std::endl
            );
        }
        else if(get_data_type() == kFixpoint){
            ICDL_ASSERT(storage_proto.data_descriptor().has_fix_point(), 
                "A protobuf object for FixpointTensorStorage Must has fixpoint data represent");
            auto r = storage_proto.data_descriptor().fix_point();
            ICDL_ASSERT(get_data_descriptor().get_represent().fix_point == FixpointRepresent().deserialize(r), 
                "FixpointRepresent not match when deserializing TensorStorage from a protobuf object"
            );
        }

        ICDL_ASSERT(storage_proto.data().size() == get_total_bytes(), "Storage size not match.");
        
        auto my_ptr = data_ptr();
        auto proto_ptr = storage_proto.data().c_str();
        memcpy(my_ptr, proto_ptr, get_total_bytes());
    }

    icdl_proto::TensorStorage TensorStorage::serialize() const{
        icdl_proto::TensorStorage s;
        s.set_data(data_ptr(), get_total_bytes());
        auto descr_pb = icdl_proto::TensorDataDescriptor();
        icdl_proto::TensorDataDescriptor_TensorDataType dtype_pb;
        switch(get_data_type()){
            case kFloat32: dtype_pb = icdl_proto::TensorDataDescriptor_TensorDataType_FLOAT_32;break;
            case kFloat16: dtype_pb = icdl_proto::TensorDataDescriptor_TensorDataType_FLOAT_16;break;
            case kFixpoint: dtype_pb = icdl_proto::TensorDataDescriptor_TensorDataType_FIXPOINT; break;
            default: 
                dtype_pb = icdl_proto::TensorDataDescriptor_TensorDataType_INVALID_DTYPE; break;
        }
        descr_pb.set_dtype(dtype_pb);

        if(get_data_type() == kFloat32 || get_data_type() == kFloat16){
            auto proto_flo = descr_pb.mutable_flo_point();
            auto storage_flo = get_data_descriptor().get_represent().flo_point;
            proto_flo->set_is_signed(storage_flo.is_signed);
            proto_flo->set_total_bits(storage_flo.total_bits);
            proto_flo->set_mantissa_bits(storage_flo.mantissa_bits);
            proto_flo->set_exp_bits(storage_flo.exp_bits);
        }
        else if(get_data_type() == kFixpoint){
            auto storage_fix = get_data_descriptor().get_represent().fix_point;
            (*descr_pb.mutable_fix_point()) = storage_fix.serialize();
        }

        (*s.mutable_data_descriptor()) = descr_pb;

        return s;
    }

    Float32TensorStorage::Float32TensorStorage(size_t num_element, TensorDataLocation data_loc)
        :TensorStorage(num_element, data_loc, Float32Descriptor()), data_ptr_(nullptr){
        if(data_loc == kCPUMem){
            data_ptr_ = new float[num_element];
        }
        else if(data_loc == kAccMem){
            void * p = accelerator_memory_malloc(num_element*sizeof(float));
            data_ptr_ = static_cast<float*>(p);
        }
        // insanity check
        if(data_ptr_ == nullptr){
            std::cerr << "Failed to allocate memory for a float Storage, exit" << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    Float32TensorStorage::Float32TensorStorage(float* blob_ptr, const size_t num_element)
        :TensorStorage(num_element, TensorDataLocation::CPU_MEMORY, Float32Descriptor()), 
        data_ptr_(blob_ptr){
        set_memory_ownership(false);
    }

    Float32TensorStorage::~Float32TensorStorage(){
        if(data_ptr_ != nullptr && own_memory_ == true){
            if(data_location_ == kCPUMem){
                delete[] data_ptr_;
            }
            else{
                accelerator_memory_dealloc(data_ptr_);
            }
        }
    }

    StoragePtr Float32TensorStorage::clone() const{
        Float32TensorStorage* fp32storage = new Float32TensorStorage(num_data_, data_location_);
        if(data_location_ == kCPUMem){
            memcpy(fp32storage->data_ptr_, data_ptr_, num_data_*sizeof(float));
        }
        else if(data_location_ == kAccMem){
            accelerator_memory_copy(fp32storage->data_ptr_, data_ptr_, num_data_*sizeof(float));
        }
        return StoragePtr(fp32storage);
    }

    FixpointTensorStorage::FixpointTensorStorage(size_t num_element, const FixpointRepresent& data_represent, TensorDataLocation data_loc)
            : TensorStorage(num_element, data_loc, data_represent), data_ptr_(nullptr){
        data_descriptor_.dtype(kFixpoint);
        if(data_loc == kCPUMem){
            size_t num_byte_per_data = data_represent.num_byte_up_round();
            data_ptr_ = new int8_t[num_element*num_byte_per_data];
        }
        else if(data_loc == kAccMem){
            void * p = accelerator_memory_malloc(data_represent, num_element); 
            data_ptr_ = static_cast<int8_t*>(p);
        }
        // insanity check
        if(data_ptr_ == nullptr){
            std::string loc_str;
            loc_str = data_loc == kCPUMem ? "CPU_MEMORY": "ACCELERATOR_MEMORY";
            std::cerr << "Failed to allocate memory for a Fixpoint Storage on" << 
            loc_str << ", exit" << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    FixpointTensorStorage::FixpointTensorStorage(int8_t* blob_ptr, const size_t num_element, const FixpointRepresent& data_represent)
        : TensorStorage(num_element, kCPUMem, data_represent), data_ptr_(blob_ptr){
        data_descriptor_.dtype(kFixpoint);
        set_memory_ownership(false);
    }

    FixpointTensorStorage::~FixpointTensorStorage(){
        if(data_ptr_ != nullptr && own_memory_ == true){
            if(data_location_ == kCPUMem){
                delete[] data_ptr_;
            }
            else{
                accelerator_memory_dealloc(data_ptr_);
            }
        }
    }

    StoragePtr FixpointTensorStorage::clone() const{
        auto fixp_represent = data_descriptor_.get_represent().fix_point;
        FixpointTensorStorage* fixpoint_storage = new FixpointTensorStorage(num_data_, 
            fixp_represent, data_location_);
            
        if(data_location_ == kCPUMem){
            size_t total_byte = num_data_ * fixp_represent.num_byte_up_round();
            //size_t total_byte = num_data_ * (data_represent_.total_bits + 8 - 1) / 8;
            memcpy(fixpoint_storage->data_ptr_, data_ptr_, total_byte);
        }
        else if(data_location_ == kAccMem){
            accelerator_memory_copy(fixpoint_storage->data_ptr_, data_ptr_, fixp_represent, num_data_);
        }
        return StoragePtr(fixpoint_storage);
    }
}//namespace icdl
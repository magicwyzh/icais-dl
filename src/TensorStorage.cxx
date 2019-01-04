#include "TensorStorage.h"
#include <string.h>
#include "tensor_utils.h"
#include "accelerator_memory.h"
namespace icdl{
    FixpointRepresent invalid_fix_represent;
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
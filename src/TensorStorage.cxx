#include "TensorStorage.h"
#include <string.h>
#include "utils.h"
#include "accelerator_memory.h"
namespace icdl{
    FixpointRepresent invalid_fix_represent;
    Float32TensorStorage::Float32TensorStorage(size_t num_element, TensorDataLocation data_loc)
        :TensorStorage(num_element, data_loc, invalid_fix_represent), data_ptr_(nullptr){
        if(data_loc == CPU_MEMORY){
            data_ptr_ = new float[num_element];
        }
        else if(data_loc == ACCELERATOR_MEMORY){
            void * p = accelerator_memory_malloc(num_element*sizeof(float));
            data_ptr_ = static_cast<float*>(p);
        }
        // insanity check
        if(data_ptr_ == nullptr){
            std::cerr << "Failed to allocate memory for a float Storage, exit" << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    Float32TensorStorage::~Float32TensorStorage(){
        if(data_ptr_ != nullptr){
            if(data_location_ == CPU_MEMORY){
                delete[] data_ptr_;
            }
            else{
                accelerator_memory_dealloc(data_ptr_);
            }
        }
    }

    StoragePtr Float32TensorStorage::clone() const{
        Float32TensorStorage* fp32storage = new Float32TensorStorage(num_data_, data_location_);
        if(data_location_ == CPU_MEMORY){
            memcpy(fp32storage->data_ptr_, data_ptr_, num_data_*sizeof(float));
        }
        else if(data_location_ == ACCELERATOR_MEMORY){
            accelerator_memory_copy(fp32storage->data_ptr_, data_ptr_, num_data_*sizeof(float));
        }
        return StoragePtr(fp32storage);
    }

    FixpointTensorStorage::FixpointTensorStorage(size_t num_element, const FixpointRepresent& data_represent, TensorDataLocation data_loc)
            : TensorStorage(num_element, data_loc, data_represent), data_ptr_(nullptr){
        if(data_loc == CPU_MEMORY){
            size_t num_byte_per_data = data_represent.num_byte_up_round();
            data_ptr_ = new uint8_t[num_element*num_byte_per_data];
        }
        else if(data_loc == ACCELERATOR_MEMORY){
            void * p = accelerator_memory_malloc(data_represent, num_element); 
            data_ptr_ = static_cast<uint8_t*>(p);
        }
        // insanity check
        if(data_ptr_ == nullptr){
            std::string loc_str;
            loc_str = data_loc == CPU_MEMORY ? "CPU_MEMORY": "ACCELERATOR_MEMORY";
            std::cerr << "Failed to allocate memory for a Fixpoint Storage on" << 
            loc_str << ", exit" << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    FixpointTensorStorage::~FixpointTensorStorage(){
        if(data_ptr_ != nullptr){
            if(data_location_ == CPU_MEMORY){
                delete[] data_ptr_;
            }
            else{
                accelerator_memory_dealloc(data_ptr_);
            }
        }
    }

    StoragePtr FixpointTensorStorage::clone() const{
        FixpointTensorStorage* fixpoint_storage = new FixpointTensorStorage(num_data_, data_represent_, data_location_);

        if(data_location_ == CPU_MEMORY){
            size_t total_byte = num_data_ * data_represent_.num_byte_up_round();
            //size_t total_byte = num_data_ * (data_represent_.total_bits + 8 - 1) / 8;
            memcpy(fixpoint_storage->data_ptr_, data_ptr_, total_byte);
        }
        else if(data_location_ == ACCELERATOR_MEMORY){
            accelerator_memory_copy(fixpoint_storage->data_ptr_, data_ptr_, data_represent_, num_data_);
        }
        return StoragePtr(fixpoint_storage);
    }
}//namespace icdl
#ifndef __ICDL_TENSOR_STORAGE_H__
#define __ICDL_TENSOR_STORAGE_H__
#include <memory>
#include <cstddef>
#include "tensor_utils.h"
#include "accelerator_memory.h"
#include <iostream>
#include <cstdlib>
namespace icdl{
    class TensorStorage;
    using StoragePtr = std::shared_ptr<TensorStorage>;

    // Though there may be different kinds of TensorStorage, we should mainly manipulate a 
    // smart pointer to the TensorStorage rather than directly use the Float32TensorStorage/FixpointTensorStorage...
    class TensorStorage {
        friend class TensorDataLoader; // to fill the storage of a tensor with either a file or other things...
        friend class StorageConverter;
    public:
        virtual void* data_ptr() const = 0;
        virtual StoragePtr clone() const = 0;
        virtual TensorDataType get_data_type() const = 0;
        size_t get_data_num() const{
            return num_data_;
        }
        TensorDataLocation get_data_location() const{
            return data_location_;
        }
        FixpointRepresent get_data_represent() const{
            return data_represent_;
        }
        // im not quite sure what this should be used.
        // but for sparse Storage this may be of some use.
        virtual void* aux_info_ptr() const{
            return aux_info_.get();
        }
        TensorStorage(size_t num_data, const TensorDataLocation& data_loc, const FixpointRepresent& data_repre): 
            num_data_(num_data), data_location_(data_loc), data_represent_(data_repre), aux_info_(nullptr), own_memory_(true){}
        bool own_memory(){
            return own_memory_;
        }
        virtual ~TensorStorage(){}
    protected:
        void set_memory_ownership(bool owned){
            own_memory_ = owned;
        }
        size_t num_data_{0}; // to indicate how many corresponding type data in the storage
        TensorDataLocation data_location_{TensorDataLocation::INVALID_LOCATION}; 
        FixpointRepresent data_represent_{0, 0, 0}; // for float, this is all set to zero
        std::shared_ptr<void> aux_info_{nullptr}; 
        // the own_memory_ means this storage should take care of memory deallocation.
        bool own_memory_{true};
    };

    // always dense! dont use aux_info
    class Float32TensorStorage: public TensorStorage{
    public:
        virtual void * data_ptr() const{
            return static_cast<void*>(data_ptr_);
        }
        virtual TensorDataType get_data_type() const override{
            return kFloat32;
        }
        virtual StoragePtr clone() const;
        //constructor
        Float32TensorStorage(size_t num_element, TensorDataLocation data_loc = CPU_MEMORY);
        //from blob
        Float32TensorStorage(float* blob_ptr, const size_t num_element);
        Float32TensorStorage(const Float32TensorStorage& rhs) = delete; //dont copy it.
        ~Float32TensorStorage();

    private:
        
        // we do not use smart pointer here mainly because the alloc/dealloc rule for CPU_Memory and Accelerator_Memory
        // maybe quite different, e.g., when the AcceleratorMemory is the FPGA's own DDR, or it is an on-chip FPGA memory, 
        // we should leave the implementation of those different kinds of memory alloc/dealloc scheme for customization. 
        // if smart pointers are used, they require the same deletor for both kinds of memory location.
        // Maybe the other day we can implement a specified Memory allocator to tackle both memory types, so that we can uniformly 
        // use the same smart pointer here. But till now i am not familiar with this and have no idea... dont do it now.
        float* data_ptr_ = nullptr;
    };

    //always dense! dont use aux_info_
    class FixpointTensorStorage: public TensorStorage{
    public:
        virtual void* data_ptr() const{
            return static_cast<void*>(data_ptr_);
        }
        virtual TensorDataType get_data_type() const override{
            return kFixpoint;
        }
        virtual StoragePtr clone() const;
        //constructor
        FixpointTensorStorage(size_t num_element, const FixpointRepresent& data_represent, TensorDataLocation data_loc = CPU_MEMORY);
        FixpointTensorStorage(const FixpointTensorStorage& rhs) = delete; // dont copy
        //from blob
        FixpointTensorStorage(int8_t* blob_ptr, const size_t num_element, const FixpointRepresent& data_represent);
        ~FixpointTensorStorage();

    private:
        int8_t* data_ptr_ = nullptr;
    };

    // The following are forwarding function to just let me not to write very long function name std::make_shared....
    template <typename... Args>
    auto f32_storage_make(Args&&... args) -> decltype(std::make_shared<Float32TensorStorage>(std::forward<Args>(args)...)) {
    return std::make_shared<Float32TensorStorage>(std::forward<Args>(args)...);
    }

    template <typename... Args>
    auto fixp_storage_make(Args&&... args) -> decltype(std::make_shared<FixpointTensorStorage>(std::forward<Args>(args)...)) {
    return std::make_shared<FixpointTensorStorage>(std::forward<Args>(args)...);
    }

}//namespace icdl
#endif

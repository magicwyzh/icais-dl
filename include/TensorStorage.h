#ifndef __ICDL_TENSOR_STORAGE_H__
#define __ICDL_TENSOR_STORAGE_H__
#include <memory>
#include <cstddef>
#include "utils.h"
#include "accelerator_memory.h"
#include <iostream>
#include <stdlib.h>
namespace icdl{
    class TensorStorage;
    typedef std::shared_ptr<TensorStorage> StoragePtr;

    // Though there may be different kinds of TensorStorage, we should mainly manipulate a 
    // smart pointer to the TensorStorage rather than directly use the Float32TensorStorage/FixpointTensorStorage...
    class TensorStorage {
        friend class TensorDataLoader; // to fill the storage of a tensor with either a file or other things...
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
        TensorStorage(size_t num_data, TensorDataLocation& data_loc, FixpointRepresent data_repre): 
            num_data_(num_data), data_location_(data_loc), data_represent_(data_repre){}
    protected:
        size_t num_data_; // to indicate how many corresponding type data in the storage
        TensorDataLocation data_location_; 
        FixpointRepresent data_represent_; // for float, this is all set to zero
    };


    class Float32TensorStorage: public TensorStorage{
    public:
        virtual void * data_ptr() const{
            return static_cast<void*>(data_ptr_);
        }
        virtual TensorDataType get_data_type() const override{
            return FLOAT_32;
        }
        virtual StoragePtr clone() const;
        //constructor
        Float32TensorStorage(size_t num_element, TensorDataLocation data_loc = CPU_MEMORY);
        Float32TensorStorage(const Float32TensorStorage& rhs) = delete; //dont copy it.
        ~Float32TensorStorage();

    private:
        
        // we do not use smart pointer here mainly because the alloc/dealloc rule for CPU_Memory and Accelerator_Memory
        // maybe quite different, e.g., when the AcceleratorMemory is the FPGA's own DDR, or it is an on-chip FPGA memory, 
        // we should leave the implementation of those different kinds of memory alloc/dealloc scheme for customization. 
        // if smart pointers are used, they require the same deletor for both kinds of memory location.
        // Maybe the other day we can implement a specified Memory allocator to tackle both memory types, so that we can uniformly 
        // use the same smart pointer here. But till now i am not familiar with this and have no idea... dont do it now.
        float* data_ptr_;
    };

    class FixpointTensorStorage: public TensorStorage{
    public:
        virtual void* data_ptr() const{
            return static_cast<void*>(data_ptr_);
        }
        virtual TensorDataType get_data_type() const override{
            return FIXPOINT;
        }
        virtual StoragePtr clone() const;
        //constructor
        FixpointTensorStorage(size_t num_element, const FixpointRepresent& data_represent, TensorDataLocation data_loc = CPU_MEMORY);
        FixpointTensorStorage(const FixpointTensorStorage& rhs) = delete; // dont copy
        ~FixpointTensorStorage();

    private:
        uint8_t* data_ptr_;
    };
}//namespace icdl
#endif

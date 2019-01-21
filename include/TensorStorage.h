#ifndef __ICDL_TENSOR_STORAGE_H__
#define __ICDL_TENSOR_STORAGE_H__
#include <memory>
#include <cstddef>
#include "tensor_utils.h"
#include "accelerator_memory.h"
#include <iostream>
#include <cstdlib>
#include "protos/Tensor.pb.h"
namespace icdl{
    class TensorStorage;
    using StoragePtr = std::shared_ptr<TensorStorage>;
    /**
     * @brief The StorageAuxInfoBase can be inherited to add some auxiliary info for a tensor storage,
     * e.g., for a sparse Tensor, indices can be saved here. This virtual base class is just for 
     * runtime polymorphism.
     */
    class StorageAuxInfoBase{
    };
    ///~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ///                         TensorStorage
    ///~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    /// Though there may be different kinds of TensorStorage, we should mainly 
    /// manipulate a  smart pointer to the TensorStorage rather than directly
    /// use the Float32TensorStorage/FixpointTensorStorage...
    ///~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ///                     Use of Raw Ptr for Data
    ///~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    /// The following reasons are taken into consideration that why we just use
    /// raw pointer to the data rather than something like smart pointer or 
    /// std::vector/std::array, etc.
    /// 1. The underlying data may come from many different sources, e.g., 
    /// the storage is created by the user, under which circumstance
    /// the storage itself should take care of mem alloc/dealloc, another example
    /// is that the underlying data is from other different data types, like 
    /// opencv::Mat, and if we want to avoid memory copy from it, the storage should
    /// not own the memory. Thus, it is not good to make it has a smart pointer that 
    /// will manage the memory.
    /// 2. Why dont use std::vector? Firstly, the std::vector may do some memory copy
    /// when the data is from ..., like opencv::Mat. It should be avoided. Secondly,
    /// for some fix-point data representation, it is possible that each element of the
    /// storage is not aligned. So i think it will not be good to use std::vector.
    /// 3. I used to consider that i can just let the constructor with blob_ptr just to copy
    /// data from the blob_ptr, which may introduce some overhead but can make the 
    /// underlying pointer to be smart pointer like std::unique_ptr, and then it maybe 
    /// possible to not take care of memory alloc/dealloc. But actually, smart pointer is 
    /// also not a good choice for array according to <<Effective Modern C++>>. Besides, when
    /// the storage is actually pointing to data in the accelerator memory, the default 
    /// memory deallocator in smart pointer is incorrect. And till now I have no idea how the 
    /// accelelrator memory should be allocated/deallocated in some FPGA SoC, and whether the
    /// CPU is able to manipulate them is not sure. I think it is quite related to the implementation
    /// of the accelerator memory. For example, when the accelerator memory is just an on-chip
    /// SRAM in the PL part of a Xilinx FPGA, then CPU has the chance to do something to it. 
    /// However, when the accelerator memory is the independent DRAM for PL part, whether CPU
    /// has access to it depends on how the DDR interface is implemented and whether the interface
    /// is also connected to the CPU or just the accelerator. Thus, it may be possible that the
    /// storage class here is just to maintain the pointer but never manipulate around it. And
    /// the customized accelerator memory alloc/dealloc interface should take care of it. 
    /// In the end, here the raw pointer is chosen for flexibility.
    ///~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    class TensorStorage {
        // to fill the storage of a tensor with either a file or other things...
        friend class TensorDataLoader; 
        friend class StorageConverter;
    public:
        // cannot copy this
        TensorStorage(const TensorStorage& rhs) = delete;
        virtual void* data_ptr() const = 0;
        virtual StoragePtr clone() const = 0;
        virtual TensorDataType get_data_type() const{
            return data_descriptor_.get_dtype();
        }
        size_t get_data_num() const{
            return num_data_;
        }
        virtual size_t get_total_bytes() const = 0;
        TensorDataLocation get_data_location() const{
            return data_location_;
        }
        // deprecated.
        FixpointRepresent get_data_represent() const{
            //return data_represent_;
            if(get_data_type()!=kFixpoint){
                std::cout<< "Warning: Try to get the fixpoint represent from a " << 
                enum_to_string(get_data_type()) << " Tensor." << std::endl;
            }
            return data_descriptor_.get_represent().fix_point;
        }
        FloatpointRepresent get_float_data_represent() const{
            if(get_data_type()!=kFloat32 || get_data_type()!=kFloat16){
                std::cout<< "Warning: Try to get the floatpoint represent from a " << 
                enum_to_string(get_data_type()) << " Tensor." << std::endl;
            }
            return data_descriptor_.get_represent().flo_point;
        }

        TensorDataDescriptor get_data_descriptor() const{
            return data_descriptor_;
        }

        // im not quite sure what this should be used.
        // but for sparse Storage this may be of some use.
        virtual std::shared_ptr<StorageAuxInfoBase> aux_info_ptr() const{
            return aux_info_;
        }

       TensorStorage(size_t num_data, const TensorDataLocation& data_loc, const FixpointRepresent& fixp_data_repre): 
            num_data_(num_data), data_location_(data_loc), 
            data_descriptor_(FixpointDescriptor(fixp_data_repre)),
            aux_info_(nullptr), own_memory_(true){}
        
        TensorStorage(size_t num_data, const TensorDataLocation& data_loc, const TensorDataDescriptor data_descriptor):
            num_data_(num_data), data_location_(data_loc), 
            data_descriptor_(data_descriptor), 
            aux_info_(nullptr), own_memory_(true){}
        bool own_memory(){
            return own_memory_;
        }
        virtual ~TensorStorage(){}

        void deserialize(const icdl_proto::TensorStorage& storage_proto);
        icdl_proto::TensorStorage serialize() const;
    protected:
        void set_memory_ownership(bool owned){
            own_memory_ = owned;
        }
        size_t num_data_{0}; // to indicate how many corresponding type data in the storage
        TensorDataLocation data_location_{TensorDataLocation::INVALID_LOCATION}; 
        TensorDataDescriptor data_descriptor_{};
        
        std::shared_ptr<StorageAuxInfoBase> aux_info_{nullptr}; 
        // the own_memory_ means this storage should take care of memory deallocation.
        bool own_memory_{true};
        
    };

    // always dense! dont use aux_info
    class Float32TensorStorage: public TensorStorage{
    public:
        virtual void * data_ptr() const{
            return static_cast<void*>(data_ptr_);
        }
        virtual size_t get_total_bytes() const override{
            return get_data_num()*sizeof(float);
        }
        virtual StoragePtr clone() const;
        //constructor
        Float32TensorStorage(size_t num_element, TensorDataLocation data_loc = kCPUMem);
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
        virtual size_t get_total_bytes() const override{
            size_t num_byte_per_data = data_descriptor_.get_represent().fix_point.num_byte_up_round();
            return num_byte_per_data * get_data_num();
        }
        virtual StoragePtr clone() const;
        //constructor
        FixpointTensorStorage(size_t num_element, const FixpointRepresent& data_represent, TensorDataLocation data_loc = kCPUMem);
        FixpointTensorStorage(const FixpointTensorStorage& rhs) = delete; // dont copy
        //from blob
        FixpointTensorStorage(int8_t* blob_ptr, const size_t num_element, const FixpointRepresent& data_represent);
        ~FixpointTensorStorage();
        
    private:
        int8_t* data_ptr_ = nullptr;
    };

    // The following are perfectly forwarding function to just let me not to write very long 
    // function name std::make_shared....
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

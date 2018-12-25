#ifndef __ICDL_TENSOR_H__
#define __ICDL_TENSOR_H__
#include "tensor_utils.h"
#include "TensorStorage.h"
#include "StorageConverter.h"
namespace icdl{

    class Tensor{
    private:
        TensorSize size_{};
        StoragePtr storage_{nullptr};
        TensorMemLayout mem_layout_{TensorMemLayout::INVALID_LAYOUT};
        TensorDataDescriptor data_descriptor_{TensorDataDescriptor()};
        OptionalTensorInfo opt_info_{OptionalTensorInfo()};

        void convert_to_fixpoint(const StorageConverter& storage_converter, 
                                 const FixpointRepresent & target_fix_represent, 
                                 const TensorMemLayout& target_mem_layout);

        void convert_to_float32(const StorageConverter& storage_converter, 
                              const TensorMemLayout& target_mem_layout);
        
    public:
        TensorDataType dtype() const;
        TensorSize size() const;
        TensorDataLocation get_data_location() const;
        TensorMemLayout get_mem_layout() const;
        TensorDataDescriptor get_data_descript() const;
        size_t nelement();
        void* data_ptr() const;
        void* aux_info_ptr() const;
        /* change data type from float-to-fixpoint or fixpoint-to-fixpoint, etc. 
        if change to fixpoint, then the fix_represent should be used.
        This will change the underlying storage. But how to change dtype is 
        up to the implementation.
        */

        // simplified ver. no layout change
        // should also be able to be used as:
        // Tensor t;
        // t.convert_to("FLOAT32");
        void convert_to(const TensorDataDescriptor& descriptor);

        void convert_to(const TensorDataDescriptor& descriptor, 
                        const TensorMemLayout& target_mem_layout);

        void convert_to(const TensorDataDescriptor& descriptor, 
                        const TensorMemLayout& target_mem_layout,
                        const StorageConverter& storage_converter);
        
        /** Constructors**/
        // a = icdl::Tensor({3,3,3}, "FLOAT32");
        // a = icdl::Tensor({3,3,3}, TensorDataDescriptor().dtype(TensorDataType::FIXPOINT).represent({8, true, 0}));
        Tensor(const TensorSize& tensor_size, 
               const TensorDataDescriptor& data_descriptor,
               const TensorDataLocation& location = TensorDataLocation::CPU_MEMORY,
               const TensorMemLayout& mem_layout = TensorMemLayout::DENSE_LAYOUT, 
               const OptionalTensorInfo optional_info = OptionalTensorInfo());
        // construct a tensor from existing blob, the storage will not own the memory.
        // by default we consider the tensor is in CPU memory and with dense layout
        // a = icdl::Tensor(image.data_ptr(), {1,3,32,32}, "FLOAT32");
        // 
        Tensor(void * blob_ptr, 
               const TensorSize& tensor_size, 
               const TensorDataDescriptor& data_descriptor,
               const TensorDataLocation& location = TensorDataLocation::CPU_MEMORY,
               const TensorMemLayout& mem_layout = TensorMemLayout::DENSE_LAYOUT,
               const OptionalTensorInfo optional_info = OptionalTensorInfo()
               );
        
        bool operator==(const Tensor& rhs) const;
        bool operator!=(const Tensor& rhs) const;
    };
}//namespace icdl


#endif

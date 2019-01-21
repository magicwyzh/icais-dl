#ifndef __ICDL_TENSOR_H__
#define __ICDL_TENSOR_H__
#include "tensor_utils.h"
#include "TensorStorage.h"
#include "StorageConverter.h"
#include "protos/Tensor.pb.h"
namespace icdl{

    class Tensor{
    private:
        TensorSize size_{};
        StoragePtr storage_{nullptr};
        TensorMemLayout mem_layout_{TensorMemLayout::INVALID_LAYOUT};
        OptionalTensorInfo opt_info_{OptionalTensorInfo()};

        Tensor convert_to_fixpoint(const StorageConverter& storage_converter, 
                                 const FixpointRepresent & target_fix_represent, 
                                 const TensorMemLayout& target_mem_layout) const;

        Tensor convert_to_float32(const StorageConverter& storage_converter, 
                              const TensorMemLayout& target_mem_layout) const;
        
        bool deserialize_storage_init(const icdl_proto::Tensor& tensor_proto);
        
    public:
        TensorDataType dtype() const;
        TensorSize size() const;
        TensorDataLocation get_data_location() const;
        TensorMemLayout get_mem_layout() const;
        TensorDataDescriptor get_data_descript() const;
        size_t nelement() const;
        void* data_ptr() const;
        std::shared_ptr<StorageAuxInfoBase> aux_info_ptr() const;
        Tensor& deserialize(const icdl_proto::Tensor& tensor_proto);
        icdl_proto::Tensor serialize() const;
        /* change data type from float-to-fixpoint or fixpoint-to-fixpoint, etc. 
        *   Retrun a new tensor, the underlying storage is changed.
        */
        // simplified ver. no layout change
        Tensor convert_to(const TensorDataDescriptor& descriptor) const;

        Tensor convert_to(const TensorDataDescriptor& descriptor, 
                        const TensorMemLayout& target_mem_layout) const;

        Tensor convert_to(const TensorDataDescriptor& descriptor, 
                        const TensorMemLayout& target_mem_layout,
                        const StorageConverter& storage_converter) const;
        
        /** Constructors**/
        // a = icdl::Tensor({3,3,3}, FixpointDescriptor(8, true, 0));
        Tensor(const TensorSize& tensor_size, 
               const TensorDataDescriptor& data_descriptor,
               const TensorDataLocation& location = kCPUMem,
               const TensorMemLayout& mem_layout = kDense, 
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
        Tensor() = default;
        Tensor(Tensor&& other) = default;
        Tensor& operator=(Tensor&& other) = default;
        Tensor(const Tensor & other) = default;
        // return a new Tensor sharing the storage with current Tensor.
        Tensor view(const TensorSize& tensor_size) const;

        bool operator==(const Tensor& rhs) const;
        bool operator!=(const Tensor& rhs) const;
    };

    using TensorList = std::vector<Tensor>;
}//namespace icdl


#endif

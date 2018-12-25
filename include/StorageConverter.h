#ifndef __ICDL_DATA_TYPE_CONVERTER_H__
#define __ICDL_DATA_TYPE_CONVERTER_H__
#include "TensorStorage.h"
#include "tensor_utils.h"
namespace icdl{
    
    // The actuall data type converter should be a subclass of it...
    class StorageConverter{
    public:
        // convert to fix_represent
        virtual StoragePtr fix_to_fix_convert(StoragePtr storage,
                                      const FixpointRepresent& src_fix_represent, 
                                      const FixpointRepresent& target_fix_represent, 
                                      const TensorMemLayout & src_mem_layout, 
                                      const TensorMemLayout &target_mem_layout) const = 0;
        virtual StoragePtr float32_to_fix_convert(StoragePtr storage,
                                        const FixpointRepresent& target_fix_represent, 
                                        const TensorMemLayout & src_mem_layout, const 
                                        TensorMemLayout &target_mem_layout) const = 0;
        // convert to float
        virtual StoragePtr fix_to_float32_convert(StoragePtr storage,
                                        const FixpointRepresent& src_fix_represent, 
                                        const TensorMemLayout & src_mem_layout, 
                                        const TensorMemLayout & target_mem_layout) const = 0;

        // only possible to change the layout
        virtual StoragePtr float32_to_float32_convert(StoragePtr storage,
                                                  const TensorMemLayout& src_mem_layout, 
                                                  const TensorMemLayout& target_mem_layout) const = 0;
    };

    class DefaultStorageConverter;

    // Singleton class
    class DefaultStorageConverter: public StorageConverter{
        friend StorageConverter& get_default_storage_converter();
    public: 
        // convert to fix_represent
        virtual StoragePtr fix_to_fix_convert(StoragePtr storage,
                                      const FixpointRepresent& src_fix_represent, 
                                      const FixpointRepresent& target_fix_represent, 
                                      const TensorMemLayout & src_mem_layout, 
                                      const TensorMemLayout &target_mem_layout) const override;
        virtual StoragePtr float32_to_fix_convert(StoragePtr storage,
                                        const FixpointRepresent& target_fix_represent, 
                                        const TensorMemLayout & src_mem_layout, const 
                                        TensorMemLayout &target_mem_layout) const override;
        // convert to float
        virtual StoragePtr fix_to_float32_convert(StoragePtr storage,
                                        const FixpointRepresent& src_fix_represent, 
                                        const TensorMemLayout & src_mem_layout, 
                                        const TensorMemLayout & target_mem_layout) const override;

        // only possible to change the layout
        virtual StoragePtr float32_to_float32_convert(StoragePtr storage,
                                                  const TensorMemLayout& src_mem_layout, 
                                                  const TensorMemLayout& target_mem_layout) const override;
        static StorageConverter& get();

    private:
        DefaultStorageConverter() = default;
    };
    

}//namespace icdl

#endif
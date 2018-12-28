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
        
        virtual ~StorageConverter(){}
    };

    class DefaultStorageConverter;

    // Singleton class
    // Get a CPUMem storage...
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
        // return a singleton storage converter
        static StorageConverter& get();
        // utils function for conversion of single data.
        // for fixpoint data, the output is int16_t so that the size is enough to contain all bits.
        static int16_t single_data_flo32_to_fixp(const float src_data, 
                                                const FixpointRepresent& dst_fix_represent);
        static int16_t single_data_fixp_to_fixp(const int16_t src_data, 
                                                const FixpointRepresent& src_fix_represent, 
                                                const FixpointRepresent& dst_fix_represent);
        static float single_data_fixp_to_flo32(const int16_t src_data, 
                                               const FixpointRepresent& src_fix_represent);
        static int16_t fixpoint_to_int16(const void* data_ptr, const FixpointRepresent& fix_repr,  const size_t bit_offset = 0);
    private:
        DefaultStorageConverter() = default;
        
    };
    

}//namespace icdl

#endif
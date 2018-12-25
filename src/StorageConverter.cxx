#include "StorageConverter.h"
namespace icdl{
    StorageConverter& get_default_storage_converter(){
        static DefaultStorageConverter cvt;
        StorageConverter& cvt_ref = cvt;
        return cvt_ref;
    }

    StorageConverter& DefaultStorageConverter::get(){
        return get_default_storage_converter();
    }


    /* Following are incorrect implementations**/
    StoragePtr DefaultStorageConverter::fix_to_float32_convert(StoragePtr storage,
                                        const FixpointRepresent& src_fix_represent, 
                                        const TensorMemLayout & src_mem_layout, 
                                        const TensorMemLayout & target_mem_layout) const{
        return storage;
    }

    StoragePtr DefaultStorageConverter::fix_to_fix_convert(StoragePtr storage,
                                      const FixpointRepresent& src_fix_represent, 
                                      const FixpointRepresent& target_fix_represent, 
                                      const TensorMemLayout & src_mem_layout, 
                                      const TensorMemLayout &target_mem_layout) const{
        return storage;
    }

    StoragePtr DefaultStorageConverter::float32_to_fix_convert(StoragePtr storage,
                                        const FixpointRepresent& target_fix_represent, 
                                        const TensorMemLayout & src_mem_layout, 
                                        const TensorMemLayout &target_mem_layout) const{
        return storage;
    }

    StoragePtr DefaultStorageConverter::float32_to_float32_convert(StoragePtr storage,
                                            const TensorMemLayout& src_mem_layout, 
                                            const TensorMemLayout& target_mem_layout) const{
        return storage;
    }
}//namespace icdl
#include "StorageConverter.h"
#include <cmath>
#include <exception>
#include <cassert>
namespace icdl{
    StorageConverter& get_default_storage_converter(){
        static DefaultStorageConverter cvt;
        StorageConverter& cvt_ref = cvt;
        return cvt_ref;
    }

    StorageConverter& DefaultStorageConverter::get(){
        return get_default_storage_converter();
    }



    static int16_t saturate(int32_t data, size_t total_bits, bool is_signed){
        if(is_signed == false){
            uint32_t max_val = (1 << total_bits) - 1;
            if(static_cast<uint32_t>(data) > max_val){
                return max_val;
            }
            else{
                return data;
            }
        }
        else{
            auto max_val = static_cast<int32_t>((1<<(total_bits-1)) - 1);
            auto min_val = -static_cast<int32_t>(1<<(total_bits-1));
            //std::cout<<"total_bits = "<<total_bits<<"max_val = " << max_val << "min_val = "<<min_val<<std::endl;
            if(data > max_val){
                return max_val;
            }
            else if(data < min_val){
                return min_val;
            }
            else {
                return data;
            }
        }
    }

    int16_t DefaultStorageConverter::single_data_flo32_to_fixp(const float src_data, 
                                        const FixpointRepresent& dst_fix_represent){
        auto total_bits = dst_fix_represent.total_bits;
        auto frac_point = dst_fix_represent.frac_point_location;
        auto is_signed = dst_fix_represent.is_signed;
        auto lsb_val = std::pow(2, -frac_point);
        auto round_int_result = saturate(std::lround(src_data / lsb_val), total_bits, is_signed);
        return round_int_result;
    }

    float DefaultStorageConverter::single_data_fixp_to_flo32(const int16_t src_data, 
                                    const FixpointRepresent& src_fix_represent){
        auto frac_point = src_fix_represent.frac_point_location;
        auto is_signed = src_fix_represent.is_signed;
        auto lsb_val = std::pow(2, -frac_point);
        if(is_signed){
            return lsb_val * src_data;
        }
        else{
            return lsb_val * static_cast<uint16_t>(src_data);
        }
    }

    int16_t DefaultStorageConverter::single_data_fixp_to_fixp(const int16_t src_data, 
                                     const FixpointRepresent& src_fix_represent, 
                                     const FixpointRepresent& dst_fix_represent){
        if(src_fix_represent.is_signed && 
           !dst_fix_represent.is_signed && 
           ((src_data >> (src_fix_represent.total_bits-1)) & 0x1)){
            //sign to unsign, and the data is less than zero
            throw std::logic_error("Try to convert a negative sigend data to an unsigned one");
        }
        int exp_diff = dst_fix_represent.frac_point_location - src_fix_represent.frac_point_location;
        int32_t shifted_data = 0;
        if(exp_diff > 0){
            shifted_data = static_cast<int32_t>(src_data) << exp_diff;
        }
        else{
            shifted_data = static_cast<int32_t>(src_data) >> (-exp_diff);
        }
        return saturate(shifted_data, dst_fix_represent.total_bits, dst_fix_represent.is_signed);
    }

    // because 16bits are enough for deep learning, so only expand a fixpoint data to 16bits for further processing, 
    // like fix2fix, fix2float.
    // the bit_offset args is for very compact data storage, e.g., 2 4-bit data in a byte. default is 0.
    // We assume little-endian!
    int16_t DefaultStorageConverter::fixpoint_to_int16(const void* data_ptr, const FixpointRepresent& fix_repr, const size_t bit_offset){
        auto data_fetch_ptr = static_cast<const int32_t*>(data_ptr);
        auto total_bits = fix_repr.total_bits;
        auto bit_mask = fix_repr.bit_mask();
        assert(total_bits <= 16);
        assert(bit_offset < 8);
        int32_t data;
        try{
            data = *data_fetch_ptr;
        }
        catch(...){
            // may be go out of range at the end of a heap memory chunk.
            return 0;
        }

        data >>= bit_offset;//eliminate the offset.
        data &= bit_mask;//mask the higher bits to zero.
        if(fix_repr.is_signed == false){
            return static_cast<int16_t>(data);
        }
        auto higher_bits_mask = ~bit_mask;
        bool is_negative = (data >> (total_bits-1))&0x1;
        if(is_negative){
            data |= higher_bits_mask;
        }
        return static_cast<int16_t>(data);
    }

    /* Following are incorrect implementations**/
    StoragePtr DefaultStorageConverter::fix_to_float32_convert(StoragePtr storage,
                                        const FixpointRepresent& src_fix_represent, 
                                        const TensorMemLayout & src_mem_layout, 
                                        const TensorMemLayout & target_mem_layout) const{
        //if not in CPUMem, may be impossible to have access to data
        assert(storage->get_data_location() == kCPUMem); 
        assert(storage->get_data_type() == kFixpoint);
        assert(src_mem_layout == kDense);
        assert(target_mem_layout == kDense);
        StoragePtr new_storage = f32_storage_make(storage->get_data_num(), kCPUMem);
        
        auto new_data_ptr = static_cast<float*>(new_storage->data_ptr());
        auto old_data_ptr = static_cast<int8_t*>(storage->data_ptr());
        try{
            for(size_t i = 0; i < storage->get_data_num();i++){
                auto num_byte_per_data = src_fix_represent.num_byte_up_round();
                auto fixp_data = fixpoint_to_int16(old_data_ptr+i*num_byte_per_data, src_fix_represent, 0);
                new_data_ptr[i] = single_data_fixp_to_flo32(fixp_data, src_fix_represent);
            }
        }
        catch(...){
            std::cerr << "Get exception when copy data for new storage during data type conversion in" 
            << __FILE__ << ": " << __LINE__<<std::endl;
            exit(EXIT_FAILURE);
        }

        return new_storage;
    }
    // should the return storage always own the memory or not?
    StoragePtr DefaultStorageConverter::fix_to_fix_convert(StoragePtr storage,
                                      const FixpointRepresent& src_fix_represent, 
                                      const FixpointRepresent& target_fix_represent, 
                                      const TensorMemLayout & src_mem_layout, 
                                      const TensorMemLayout &target_mem_layout) const{
        assert(src_mem_layout == kDense);
        assert(target_mem_layout == kDense);
        //if not in CPUMem, may be impossible to have access to data
        assert(storage->get_data_location() == kCPUMem); 
        assert(storage->get_data_type() == kFixpoint);

        auto num_byte_per_data_dst = target_fix_represent.num_byte_up_round();
        //StoragePtr new_storage = std::make_shared<FixpointTensorStorage>(storage->get_data_num(), target_fix_represent, kCPUMem);
        StoragePtr new_storage = fixp_storage_make(storage->get_data_num(), target_fix_represent, kCPUMem);
        
        auto old_data_ptr = static_cast<int8_t*>(storage->data_ptr());
        auto new_data_ptr = static_cast<int8_t*>(new_storage->data_ptr());
        try{
            for(size_t i = 0; i < storage->get_data_num(); i++){
                auto fixp_data = fixpoint_to_int16(old_data_ptr, src_fix_represent, 0);
                auto converted_data = single_data_fixp_to_fixp(fixp_data, src_fix_represent, target_fix_represent);
                *new_data_ptr = static_cast<int8_t>(converted_data);
                new_data_ptr++;
                if(num_byte_per_data_dst == 2){
                    *new_data_ptr = static_cast<int8_t>(converted_data>>8);
                    new_data_ptr++;
                }
                old_data_ptr += src_fix_represent.num_byte_up_round();
            }
        }
        catch(...){
            std::cerr << "Get exception when copy data for new storage during data type conversion in " 
            << __FILE__ << ": " << __LINE__<<std::endl;
            exit(EXIT_FAILURE);
        }
        return new_storage;
    }

    StoragePtr DefaultStorageConverter::float32_to_fix_convert(StoragePtr storage,
                                        const FixpointRepresent& target_fix_represent, 
                                        const TensorMemLayout & src_mem_layout, 
                                        const TensorMemLayout &target_mem_layout) const{
        assert(src_mem_layout == kDense);
        assert(target_mem_layout == kDense);
        assert(storage->get_data_location() == kCPUMem); 
        assert(storage->get_data_type() == kFloat32);
        
        auto num_byte_per_data_dst = target_fix_represent.num_byte_up_round();
        assert(num_byte_per_data_dst<=2);
        auto new_storage = fixp_storage_make(storage->get_data_num(), target_fix_represent, kCPUMem);
        auto data_ptr_8b = static_cast<int8_t*>(new_storage->data_ptr());
        auto data_ptr_16b = static_cast<int16_t*>(new_storage->data_ptr());
        auto data_ptr_old_storage = static_cast<float*>(storage->data_ptr());
        for(size_t i = 0; i < storage->get_data_num();i++){
            if(num_byte_per_data_dst==1){
                data_ptr_8b[i] = static_cast<int8_t>(single_data_flo32_to_fixp(data_ptr_old_storage[i], target_fix_represent));
            }
            else{
                data_ptr_16b[i] = single_data_flo32_to_fixp(data_ptr_old_storage[i], target_fix_represent);
            }
        }
        return new_storage;
    }

    StoragePtr DefaultStorageConverter::float32_to_float32_convert(StoragePtr storage,
                                            const TensorMemLayout& src_mem_layout, 
                                            const TensorMemLayout& target_mem_layout) const{
        if(src_mem_layout == target_mem_layout){
            return storage;
        }
        else{
            throw std::runtime_error("float32_to_float32 with different mem layout has not been implemented");
        }
    }

}//namespace icdl
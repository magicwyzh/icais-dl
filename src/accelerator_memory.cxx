#include "utils.h"
#include <cstddef>
#include <cstring>
#include <cstdint>
namespace icdl{
    void * accelerator_memory_malloc(size_t num_of_byte){
        //currently just use the cpu memory
        uint8_t* ptr = new uint8_t[num_of_byte];
        return static_cast<void*>(ptr);
    }
    void * accelerator_memory_malloc(const FixpointRepresent& data_represent, size_t num_data){
        //currently just use the cpu memory
        size_t num_byte_per_data = (data_represent.total_bits + 7) / 8;
        uint8_t* ptr = new uint8_t[num_byte_per_data*num_data];
        return static_cast<void*>(ptr);
    }



    void accelerator_memory_copy(void * dst, void * src, size_t num_byte){
        //currently just work as cpu memory
        std::memcpy(dst, src, num_byte);
    }
    void accelerator_memory_copy(void * dst, const void * src, const FixpointRepresent& data_represent, const size_t num_data){
        //currently just work as cpu memory
        size_t num_byte_per_data = (data_represent.total_bits + 7) / 8;
        std::memcpy(dst, src, num_data*num_byte_per_data);
    }

}//namespace icdl
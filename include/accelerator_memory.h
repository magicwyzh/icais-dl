#ifndef __ICDL_ACCELERATOR_MEMORY_H__
#define __ICDL_ACCELERATOR_MEMORY_H__
#include "utils.h"
#include <cstddef>
#include <cstring>
namespace icdl{
    void * accelerator_memory_malloc(size_t num_of_byte);
    void * accelerator_memory_malloc(const FixpointRepresent& data_represent, size_t num_data);

    template<typename T>
    void accelerator_memory_dealloc(T* ptr){
        delete [] ptr;
    }

    void accelerator_memory_copy(void * dst, void * src, size_t num_byte);
    void accelerator_memory_copy(void * dst, const void * src, const FixpointRepresent& data_represent, const size_t num_data);

}//namespace icdl
#endif
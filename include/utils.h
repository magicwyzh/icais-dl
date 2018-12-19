#ifndef __ICDL_TENSOR_UTILS_H__
#define __ICDL_TENSOR_UTILS_H__
#include <cstddef>
using namespace std;
namespace icdl{
    typedef enum {
        CPU_MEMORY,
        ACCELERATOR_MEMORY
    } TensorDataLocation;

    typedef enum {
        FLOAT_32,
        FIXPOINT
    } TensorDataType;

    struct FixpointRepresent{
        size_t total_bits;
        bool is_signed;
        int frac_point_location;
        FixpointRepresent():total_bits(0), is_signed(false), frac_point_location(0){}
        FixpointRepresent(size_t bits, bool sign, int frac_loc)
        :total_bits(bits), is_signed(sign), frac_point_location(frac_loc){}
        bool operator==(const FixpointRepresent& rhs) const{
            if(total_bits==rhs.total_bits && is_signed == rhs.is_signed && frac_point_location == rhs.frac_point_location){
                return true;
            }
            else {
                return false;
            }
        }
        size_t num_byte_up_round() const{
            return (total_bits + 8 - 1) / 8;
        }
    };
}
#endif
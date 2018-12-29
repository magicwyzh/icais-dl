#ifndef __ICDL_TENSOR_UTILS_H__
#define __ICDL_TENSOR_UTILS_H__
#include <cstddef>
#include <vector>
#include <cstdlib>
#include <algorithm>  
#include <cassert>
#include <string>
namespace icdl{
    using TensorSize = std::vector<size_t>;

    enum class TensorDataLocation{
        CPU_MEMORY,
        ACCELERATOR_MEMORY,
        INVALID_LOCATION
    } ;

    enum class TensorMemLayout{
        DENSE_LAYOUT,
        SPARSE_LAYOUT,
        INVALID_LAYOUT
    };

    //FLOAT_32, FLOAT_16, FIXPOINT
    enum class TensorDataType{
        FLOAT_32,
        FLOAT_16,
        FIXPOINT,
        INVALID_DTYPE
    };

    enum class TensorDimLayout4D{
        NCHW,
        NHWC,
        NOT_AVAILABLE
    };
    const std::string enum_to_string(const TensorDataType& enum_val);
    const std::string enum_to_string(const TensorDataLocation& enum_val);
    const std::string enum_to_string(const TensorMemLayout& enum_val);
    struct FixpointRepresent{
        size_t total_bits = 0;
        bool is_signed = false;
        int frac_point_location = 0;
        inline int32_t bit_mask() const{
            return (1 << total_bits) - 1;
        }
        FixpointRepresent():total_bits(0), is_signed(false), frac_point_location(0){}
        FixpointRepresent(size_t bits, bool sign, int frac_loc)
        :total_bits(bits), is_signed(sign), frac_point_location(frac_loc){}
        bool operator==(const FixpointRepresent& rhs) const;
        size_t num_byte_up_round() const{
            return (total_bits + 8 - 1) / 8;
        }
    };

    struct FloatpointRepresent{
        size_t total_bits{32};
        bool is_signed{true};
        size_t exp_bits{8};
        size_t mantissa_bits{23};
        FloatpointRepresent(): total_bits(32), is_signed(true), exp_bits(8), mantissa_bits(23){}
        FloatpointRepresent(bool fp16): total_bits(16), is_signed(true), exp_bits(5), mantissa_bits(11){}
        FloatpointRepresent(size_t _total_bits, size_t _is_signed, size_t _exp_bits, size_t _mantissa_bits):
            total_bits(_total_bits), is_signed(_is_signed), exp_bits(_exp_bits), mantissa_bits(_mantissa_bits){}
        bool operator==(const FloatpointRepresent& rhs) const;
    };

    union DataRepresent{
        FloatpointRepresent flo_point;
        FixpointRepresent fix_point;
        DataRepresent(){}
    };

    class TensorDataDescriptor{
    private:
        TensorDataType dtype_{TensorDataType::INVALID_DTYPE};
        DataRepresent represent_;
    public:
        TensorDataDescriptor(const FloatpointRepresent& float_represent);

        TensorDataDescriptor(const FixpointRepresent& fix_represent):dtype_(TensorDataType::FIXPOINT){
            represent_.fix_point = fix_represent;
        }
        //Common Float by string
        //TensorDataDescriptor(const std::string & type_str);

        // Fixpoint
        TensorDataDescriptor(const size_t total_bits, const bool is_signed, const int frac_point);
        // invalid empty descriptor
        TensorDataDescriptor();

        TensorDataType get_dtype() const;
        DataRepresent get_represent() const;
        TensorDataDescriptor& dtype(const TensorDataType& type_of_data);
        TensorDataDescriptor& represent(const FloatpointRepresent& float_represent);
        TensorDataDescriptor& represent(const FixpointRepresent& fix_represent);

        bool operator==(const TensorDataDescriptor & rhs) const;
    };
    
    struct OptionalTensorInfo{
        TensorDimLayout4D dim_layout_4d;
        OptionalTensorInfo(): dim_layout_4d(TensorDimLayout4D::NOT_AVAILABLE){}
        void set_dim_layout_4d(const TensorDimLayout4D& v){
            dim_layout_4d = v;
        }
        bool operator==(const OptionalTensorInfo& rhs) const{
            return this->dim_layout_4d==rhs.dim_layout_4d ? true : false;
        }
    };
    constexpr auto kFloat32 = TensorDataType::FLOAT_32;
    constexpr auto kFloat16 = TensorDataType::FLOAT_16;
    constexpr auto kFixpoint = TensorDataType::FIXPOINT;
    constexpr auto kCPUMem = TensorDataLocation::CPU_MEMORY;
    constexpr auto kAccMem = TensorDataLocation::ACCELERATOR_MEMORY;
    constexpr auto kDense = TensorMemLayout::DENSE_LAYOUT;
    constexpr auto kSparse = TensorMemLayout::SPARSE_LAYOUT;
    constexpr auto kNCHW = TensorDimLayout4D::NCHW;
    constexpr auto kNHWC = TensorDimLayout4D::NHWC;

    inline TensorDataDescriptor Float32Descriptor(){
        return TensorDataDescriptor().dtype(kFloat32);
    }
    inline TensorDataDescriptor Float16Descriptor(){
        return TensorDataDescriptor().dtype(kFloat16);
    }
    inline TensorDataDescriptor FixpointDescriptor(const size_t total_bits, const bool is_signed, const int frac_point){
        return TensorDataDescriptor(total_bits, is_signed, frac_point);
    }
    inline TensorDataDescriptor FixpointDescriptor(const FixpointRepresent& fix_represent){
        return TensorDataDescriptor(fix_represent);
    }
}
#endif
#ifndef __ICDL_TENSOR_UTILS_H__
#define __ICDL_TENSOR_UTILS_H__
#include <cstddef>
#include <vector>
#include <cstdlib>
#include <algorithm>  
#include <cassert>
#include <string>
#include "icdl_exceptions.h"
#include "protos/Tensor.pb.h"
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

    /**
     * @brief 
     * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
     *                      Fixpoint Represent Design Considerations
     * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
     * There may be different understandings about fixpoint representation or quantization.
     * 1. Quantization is just to make data fixpoint. E.g., a float point number 3.246 may be 
     * quantized to 3 decimal digit and has 2 fractions, then it become 3.25. And the data representation
     * of this number includes three parts: 
     *      1) total number of digit=3 
     *      2) signed or unsigned
     *      3) the fixed number has 2 fractional bits in the rightest part.
     * 2. Quantization to integer number(no fixed point info) with scalars. When one want to quantize a data x, 
     * they just represent it as x = scalar*(xq-z), where scalar is a float and xq & z are integer without fractions
     * so the data representation of this number includes x parts:
     *      1) total number of integer bits for xq & z
     *      2) signed or unsigned
     *      3) float scalar number
     * Another concern is the quantization may be per-layer quantization or per-channel or from other dims.
     *  So the info/represent mentioned above should be variable length(i.e., using std::vector).
     * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
     */
    struct FixpointRepresent{
        std::vector<uint8_t> total_bits;
        std::vector<bool> is_signed;
        std::vector<int8_t> frac_point_locations;
        std::vector<float> scalars;
        std::vector<int16_t> zero_points;
        FixpointRepresent(){}
        FixpointRepresent(uint8_t bits, bool sign, int8_t frac_loc)// per layer fixpoint without scalar
          :total_bits({bits}), is_signed({sign}), frac_point_locations({frac_loc}){}
        FixpointRepresent(std::vector<uint8_t> bits, std::vector<bool> sign, std::vector<int8_t> frac_loc)// fixpoint without scalar
          :total_bits(bits), is_signed(sign), frac_point_locations(frac_loc), scalars({}), zero_points({}){}
        FixpointRepresent(std::vector<uint8_t> bits, std::vector<bool> sign, std::vector<float> scalar, std::vector<int16_t> zero)
          :total_bits(bits), is_signed(sign), frac_point_locations({}), scalars(scalar), zero_points(zero){}
        FixpointRepresent(std::vector<uint8_t> bits, std::vector<bool> sign, 
                            std::vector<int8_t> frac_loc, std::vector<float> scalar, 
                            std::vector<int16_t> zero)
          :total_bits(bits), is_signed(sign), frac_point_locations(frac_loc), scalars(scalar), zero_points(zero){}
        FixpointRepresent(FixpointRepresent&& other) = default;
        FixpointRepresent(const FixpointRepresent&) = default;
        FixpointRepresent& deserialize(const icdl_proto::FixpointRepresent& proto_repr);
        icdl_proto::FixpointRepresent serialize() const;
        FixpointRepresent& operator=(const FixpointRepresent& other) = default;
        bool operator==(const FixpointRepresent& rhs) const;

        /**
         * @brief Return bit mask for the first element in total_bits. 
         *        The size of total bits is assumed to be > 1, otherwise raise asssertion error.
         * 
         * @return int32_t 
         */
        int32_t bit_mask() const;
        /**
         * @brief Return bit masks for each element in total_bits
         * 
         * @return std::vector<int32_t> 
         */
        std::vector<int32_t> bit_masks() const;
        /**
         * @brief Returns the number of bytes of a data, aligned to 8 bits.
         *        Assume the total_bits are all the same. Only check the first one in the total_bits.
         * 
         * @return size_t
         */
        size_t num_byte_up_round() const;
        /**
         * @brief Clear all members.
         * 
         */
        void clear();
    };


    struct FloatpointRepresent{
        size_t total_bits{32};
        bool is_signed{true};
        size_t exp_bits{8};
        size_t mantissa_bits{23};
        FloatpointRepresent(): total_bits(32), is_signed(true), exp_bits(8), mantissa_bits(23){}
        FloatpointRepresent(bool fp16);
        FloatpointRepresent(size_t _total_bits, size_t _is_signed, size_t _exp_bits, size_t _mantissa_bits):
            total_bits(_total_bits), is_signed(_is_signed), exp_bits(_exp_bits), mantissa_bits(_mantissa_bits){}
        bool operator==(const FloatpointRepresent& rhs) const;
    };

    /**
     * @brief A structure to store how data is represented as fixpoint or float.
     *        Previously it is considered to be an union, however, when the fixpoint represent
     *        starts to store multiple vectors, an union is unable to be used because all of its
     *        default constructors/destructors/copy constructors are deleted...
     *        Not sure how to write the copy constructor, so dont use union now.
     *        
     */
    struct DataRepresent{
        FixpointRepresent fix_point;
        FloatpointRepresent flo_point;
        DataRepresent(DataRepresent && other) = default;
        DataRepresent(const DataRepresent& other): fix_point(other.fix_point), flo_point(other.flo_point){}
        DataRepresent() = default;
        DataRepresent& operator=(const DataRepresent& other) = default;
    };

    class TensorDataDescriptor{
    private:
        TensorDataType dtype_{TensorDataType::INVALID_DTYPE};
        DataRepresent represent_;
    public:
        TensorDataDescriptor(const FloatpointRepresent& float_represent);
        TensorDataDescriptor(const FixpointRepresent& fix_represent);
        TensorDataDescriptor(const TensorDataDescriptor& other);
        // Fixpoint
        TensorDataDescriptor(const size_t total_bits, const bool is_signed, const int frac_point);
        // invalid empty descriptor
        TensorDataDescriptor();
        TensorDataDescriptor& operator=(const TensorDataDescriptor& other) = default;

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

    inline TensorDataDescriptor FixpointDescriptor(std::vector<uint8_t> bits, 
                                                   std::vector<bool> sign, 
                                                   std::vector<float> scalar, 
                                                   std::vector<int16_t> zero
                                                   ){
        return TensorDataDescriptor(FixpointDescriptor(bits, sign, scalar,zero));
    }


}
#endif
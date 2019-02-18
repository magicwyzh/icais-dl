#include "tensor_utils.h"
#include <iostream>
namespace icdl{
    const std::string enum_to_string(const TensorDataType& enum_val){
        switch(enum_val){
            case kFloat32: return "FLOAT_32"; break;
            case kFloat16: return "FLOAT_16"; break;
            case kFixpoint: return "FIXPOINT"; break;
            case TensorDataType::INVALID_DTYPE: return "INVALID_DTYPE"; break;
            default: return "Unknown";
        }
    }
    const std::string enum_to_string(const TensorDataLocation& enum_val){
        switch(enum_val){
            case kCPUMem: return "CPU_MEMORY"; break;
            case kAccMem: return "ACCELERATOR_MEMORY"; break;
            case TensorDataLocation::INVALID_LOCATION: return "INVALID_LOCATION"; break;
            default: return "Unknown";
        }
    }
    const std::string enum_to_string(const TensorMemLayout& enum_val){
        switch(enum_val){
            case kDense: return "DENSE_LAYOUT"; break;
            case kSparse: return "SPARSE_LAYOUT"; break;
            case TensorMemLayout::INVALID_LAYOUT: return "INVALID_LAYOUT"; break;
            default: return "Unknown";
        }
    }
    
    bool FixpointRepresent::operator==(const FixpointRepresent& rhs) const{
        if(total_bits==rhs.total_bits && 
            is_signed == rhs.is_signed && 
            frac_point_locations == rhs.frac_point_locations &&
            zero_points == rhs.zero_points &&
            scalars == rhs.scalars){
            return true;
        }
        else {
            return false;
        }
    }
    FloatpointRepresent::FloatpointRepresent(bool fp16): total_bits(16), is_signed(true), exp_bits(5), mantissa_bits(11){
        if(fp16 == false){
            total_bits = 32;
            exp_bits = 8;
            mantissa_bits = 23;
        }
    }

    bool FloatpointRepresent::operator==(const FloatpointRepresent& rhs) const{
        if( total_bits==rhs.total_bits && 
            is_signed == rhs.is_signed && 
            exp_bits == rhs.exp_bits &&
            mantissa_bits == rhs.mantissa_bits){
            return true;
        }
        else {
            return false;
        }
    }


    TensorDataDescriptor& TensorDataDescriptor::dtype(const TensorDataType& type_of_data){
        dtype_ = type_of_data;
        if(dtype_ != kFixpoint){
            represent_.flo_point = FloatpointRepresent(dtype_ == kFloat16);
        }
        return *this;
    }

    TensorDataDescriptor& TensorDataDescriptor::represent(const FloatpointRepresent& float_represent){
        assert(dtype_ == TensorDataType::FLOAT_32 || dtype_ == TensorDataType::FLOAT_16);
        represent_.flo_point = float_represent;
        return *this;
    }

    TensorDataDescriptor& TensorDataDescriptor::represent(const FixpointRepresent& fix_represent){
        assert(dtype_ == kFixpoint);
        represent_.fix_point = fix_represent;
        return *this;
    }
    DataRepresent TensorDataDescriptor::get_represent() const{
        return represent_;
    }

    TensorDataType TensorDataDescriptor::get_dtype() const{
        return dtype_;
    }

    TensorDataDescriptor::TensorDataDescriptor(const FloatpointRepresent& float_represent){
        represent_.flo_point = float_represent;
        if(float_represent.total_bits == 16){
            dtype_ = TensorDataType::FLOAT_16;
            std::cerr << "FLOAT_16 is not supported now!" << std::endl;
            exit(EXIT_FAILURE);
        }
        else if(float_represent.total_bits == 32){
            dtype_ = TensorDataType::FLOAT_32;
        }
        else{
            std::cerr << "Invalid FloatpointRepresent" << std::endl;
            exit(EXIT_FAILURE);
        }
    }
    TensorDataDescriptor::TensorDataDescriptor(): dtype_(TensorDataType::INVALID_DTYPE){}

    TensorDataDescriptor::TensorDataDescriptor(const FixpointRepresent& fix_represent):dtype_(TensorDataType::FIXPOINT){
        represent_.fix_point = fix_represent;
    }


    TensorDataDescriptor::TensorDataDescriptor(const TensorDataDescriptor& other): dtype_(other.dtype_){
        if(other.dtype_ == kFixpoint){
            represent_.fix_point = other.represent_.fix_point;
        }
        else if(other.dtype_ == kFloat32 || other.dtype_ == kFloat16){
            represent_.flo_point = other.represent_.flo_point;
        }
    }

    TensorDataDescriptor::TensorDataDescriptor(const size_t total_bits, const bool is_signed, const int frac_point){
        dtype_ = TensorDataType::FIXPOINT;
        represent_.fix_point = FixpointRepresent(total_bits, is_signed,frac_point);
    }

    bool TensorDataDescriptor::operator==(const TensorDataDescriptor & rhs) const{
        if(dtype_ != rhs.dtype_){
            return false;
        }
        if(dtype_ == kFloat32 || dtype_ == kFloat16){
            return represent_.flo_point == rhs.represent_.flo_point;
        }
        else if(dtype_ == kFixpoint){
            return represent_.fix_point == rhs.represent_.fix_point;
        }
        else{
            // INVALID dtype
            return true;
        }
    }

    void FixpointRepresent::clear(){
        total_bits.clear();
        is_signed.clear();
        frac_point_locations.clear();
        scalars.clear();
        zero_points.clear();
    }
    std::vector<int32_t> FixpointRepresent::bit_masks() const{
        std::vector<int32_t> vec;
        for(const auto& x : total_bits){
            vec.emplace_back((1<<x) - 1);
        }
        return vec;
    }
    int32_t FixpointRepresent::bit_mask() const{
        ICDL_ASSERT(total_bits.size() > 0, "total_bits field of FixpointRepresent should has more than one value for bit_mask!");
        return (1 << total_bits[0]) - 1;
    }
    size_t FixpointRepresent::num_byte_up_round() const{
        ICDL_ASSERT(total_bits.size() > 0, "total_bits field of FixpointRepresent should has more than one value for num_byte_up_round!");
        return (total_bits[0] + 8 - 1) / 8;
    }

    FixpointRepresent& FixpointRepresent::deserialize(const icdl_proto::FixpointRepresent& proto_repr){
        clear();
        for(const auto x: proto_repr.total_bits()){
            total_bits.emplace_back(static_cast<uint8_t>(x));
        }
        for(const auto x: proto_repr.is_signed()){
            is_signed.emplace_back(x);
        }
        for(const auto x: proto_repr.frac_point_locations()){
            frac_point_locations.emplace_back(static_cast<int8_t>(x));
        }
        for(const auto x: proto_repr.scalars()){
            scalars.emplace_back(static_cast<float>(x));
        }
        for(const auto x: proto_repr.zero_points()){
            zero_points.emplace_back(static_cast<int16_t>(x));
        }
        return *this;
    }
    icdl_proto::FixpointRepresent FixpointRepresent::serialize() const{
        icdl_proto::FixpointRepresent repr_pb;
        for(const auto x : frac_point_locations){
            repr_pb.add_frac_point_locations(x);
        }
        for(const auto x : is_signed){
            repr_pb.add_is_signed(x);
        }
        for(const auto x : total_bits){
            repr_pb.add_total_bits(x);
        }
        for(const auto x : scalars){
            repr_pb.add_scalars(x);
        }
        for(const auto x : zero_points){
            repr_pb.add_zero_points(x);
        }
        return repr_pb;
    }
}//namespace icdl
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
        if(total_bits==rhs.total_bits && is_signed == rhs.is_signed && frac_point_location == rhs.frac_point_location){
            return true;
        }
        else {
            return false;
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
        return *this;
    }

    TensorDataDescriptor& TensorDataDescriptor::represent(const FloatpointRepresent& float_represent){
        assert(dtype_ == TensorDataType::FLOAT_32 || dtype_ == TensorDataType::FLOAT_16);
        represent_.flo_point = float_represent;
        return *this;
    }

    TensorDataDescriptor& TensorDataDescriptor::represent(const FixpointRepresent& fix_represent){
        assert(dtype_ == TensorDataType::FIXPOINT);
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

    /* dont use this, will be confusing when constructing tensor
    TensorDataDescriptor::TensorDataDescriptor(const std::string & type_str){
        std::transform(type_str.begin(), type_str.end(), type_str.begin(), ::toupper);
        assert(type_str == "FLOAT32" || type_str == "FLOAT16");
        if(type_str == "FLOAT32"){
            dtype_ = TensorDataType::FLOAT_32;
            represent_.flo_point = FloatpointRepresent();
        }
        else{
            dtype_ = TensorDataType::FLOAT_16;
            represent_.flo_point = FloatpointRepresent(true);
            std::cerr << "FLOAT_16 is not supported now!" << std::endl;
            exit(EXIT_FAILURE);
        }
    }
    */
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
}//namespace icdl
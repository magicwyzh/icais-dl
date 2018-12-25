#include "Tensor.h"
#include "tensor_utils.h"
#include <cassert>
namespace icdl{
    TensorDataType Tensor::dtype() const{
        if(storage_ == nullptr){
            return TensorDataType::INVALID_DTYPE;
        }
            return storage_->get_data_type();
    }

    TensorSize Tensor::size() const{
        return size_;
    }

    TensorDataLocation Tensor::get_data_location() const{
        if(storage_ == nullptr){
            return TensorDataLocation::INVALID_LOCATION;
        }
        return storage_->get_data_location();
    }

    TensorMemLayout Tensor::get_mem_layout() const{
        if(storage_ == nullptr){
            return TensorMemLayout::INVALID_LAYOUT;
        }
        return mem_layout_;
    }

    TensorDataDescriptor Tensor::get_data_descript() const{
        return data_descriptor_;
    }

    void* Tensor::data_ptr() const{
        if(storage_ == nullptr){
            return nullptr;
        }
        return storage_->data_ptr();
    }

    void* Tensor::aux_info_ptr() const{
        if(storage_ == nullptr){
            return nullptr;
        }
        return storage_->aux_info_ptr();
    }

    void Tensor::convert_to_fixpoint(const StorageConverter& storage_converter,const FixpointRepresent & target_fix_represent,  const TensorMemLayout& target_mem_layout){
        if(this->dtype() == TensorDataType::FIXPOINT){
            storage_ =  storage_converter.fix_to_fix_convert(
                storage_,
                this->storage_->get_data_represent(),//src_fix_represent
                target_fix_represent, //target_fix_represent
                this->get_mem_layout(),//src_mem_layout
                target_mem_layout
            );
        }
        else if(this->dtype() == TensorDataType::FLOAT_32){
            storage_ = storage_converter.float32_to_fix_convert(
                storage_,
                target_fix_represent, 
                this->get_mem_layout(),
                target_mem_layout
            );
        }
        else{
            std::cerr << "Try to Convert an INVALID Tensor to Fixpoint" << std::endl;
            exit(EXIT_FAILURE);
        }
        // post-processing some info and check
        assert(storage_->get_data_type() == TensorDataType::FIXPOINT);
        this->mem_layout_ = target_mem_layout;
        assert(storage_->get_data_represent() == target_fix_represent);
        
    }

    void Tensor::convert_to_float32(const StorageConverter& storage_converter, const TensorMemLayout& target_mem_layout){
        if(this->dtype() == TensorDataType::FIXPOINT){
            storage_ = storage_converter.fix_to_float32_convert(
                storage_,
                this->storage_->get_data_represent(), //src_fix_represent
                this->get_mem_layout(), 
                target_mem_layout
            );
        }
        else if(this->dtype() == TensorDataType::FLOAT_32){
            storage_ = storage_converter.float32_to_float32_convert(
                storage_,
                this->get_mem_layout(),
                target_mem_layout
            );
        }
        else{
            std::cerr << "Try to Convert an INVALID Tensor to Floatpoint" << std::endl;
            exit(EXIT_FAILURE);
        }
        this->mem_layout_ = target_mem_layout;
        assert(storage_->get_data_type() == TensorDataType::FLOAT_32);
    }

    void Tensor::convert_to(const TensorDataDescriptor& descriptor, const TensorMemLayout& target_mem_layout, const StorageConverter& storage_converter){
        if(descriptor.get_dtype() == TensorDataType::FLOAT_32){
            convert_to_float32(
                storage_converter,
                target_mem_layout
            );
        }
        else if(descriptor.get_dtype() == TensorDataType::FIXPOINT){
            convert_to_fixpoint(
                storage_converter,
                descriptor.get_represent().fix_point,
                target_mem_layout
            );
        }
        else{
            if(descriptor.get_dtype() == TensorDataType::FLOAT_16){
                std::cerr << "FLOAT16 is not supported now. Dont convert to FLOAT16!" << std::endl;
            }
            std::cerr << "Invalid TensorDataType for conversion!" << std::endl;
            exit(EXIT_FAILURE);
        }
        assert(storage_->get_data_type() == descriptor.get_dtype());
    }

    void Tensor::convert_to(const TensorDataDescriptor& descriptor){
        convert_to(descriptor, mem_layout_);
    }

    void Tensor::convert_to(const TensorDataDescriptor& descriptor, const TensorMemLayout& target_mem_layout){
        convert_to(descriptor, target_mem_layout, DefaultStorageConverter::get());
    }
    
    size_t Tensor::nelement(){
        if(size_.size() == 0){
            return 0;
        }
        size_t num_element = 1;
        for(auto dim : size_){
            num_element *= dim;
        }
        return num_element;
    }

    Tensor::Tensor(const TensorSize& tensor_size, 
                   const TensorDataDescriptor& data_descriptor,
                   const TensorDataLocation& location,
                   const TensorMemLayout& mem_layout,
                   const OptionalTensorInfo optional_info)
        : size_(tensor_size), mem_layout_(mem_layout), data_descriptor_(data_descriptor),  opt_info_(optional_info){
        
        if(mem_layout_ != kDense){
            std::cerr << "Try to create a Tensor with invalid memory layout: " << enum_to_string(mem_layout) << ". Only Dense layout is supported now!" << std::endl;
            exit(EXIT_FAILURE);
        }

        size_t num_element = nelement();
        auto dtype = data_descriptor.get_dtype();
        // storage gen
        switch(dtype){
            case kFloat32:
                storage_ = f32_storage_make(num_element, location);
                break;
            case kFixpoint:
                storage_ = fixp_storage_make(num_element, data_descriptor.get_represent().fix_point, location);
                break;
            case kFloat16:
                std::cerr << "Float16 is not supported now!" << std::endl;
            default:
                std::cerr << "Try to create a Tensor with invalid data type: " << enum_to_string(data_descriptor.get_dtype()) << std::endl;
                exit(EXIT_FAILURE);
                break;
        }
        // insanity check
        assert(storage_->own_memory() == true);
    }

    Tensor::Tensor(void * blob_ptr, 
               const TensorSize& tensor_size, 
               const TensorDataDescriptor& data_descriptor,
               const TensorDataLocation& location,
               const TensorMemLayout& mem_layout,
               const OptionalTensorInfo optional_info)
        : size_(tensor_size),  mem_layout_(mem_layout), data_descriptor_(data_descriptor), opt_info_(optional_info){
        
        if(mem_layout_ != kDense){
            std::cerr << "Try to create a Tensor with invalid memory layout: " << enum_to_string(mem_layout) << ". Only Dense layout is supported now!" << std::endl;
            exit(EXIT_FAILURE);
        }
        auto dtype = data_descriptor.get_dtype();
        switch(dtype){
            case kFloat32:
                storage_ = f32_storage_make(static_cast<float*>(blob_ptr), nelement());
                break;
            case kFixpoint:
                storage_ = fixp_storage_make(static_cast<int8_t*>(blob_ptr), nelement(), data_descriptor.get_represent().fix_point);
                break;
            case kFloat16:
                std::cerr << "Float16 is not supported now!" << std::endl;
            default:
                std::cerr << "Try to create a Tensor with invalid data type from blob: " << enum_to_string(data_descriptor.get_dtype()) << std::endl;
                exit(EXIT_FAILURE);
                break;
        }
        assert(storage_->own_memory() == false);
    }

    bool Tensor::operator==(const Tensor& rhs) const{
        if(size_==rhs.size_ &&
            storage_ == rhs.storage_&&
            mem_layout_ == rhs.mem_layout_&&
            data_descriptor_ == rhs.data_descriptor_ &&
            opt_info_ == rhs.opt_info_
        ){
            return true;
        }
        else {
            return false;
        }
    }

    bool Tensor::operator!=(const Tensor& rhs) const{
        return !operator==(rhs);
    }
}//namespace icdl
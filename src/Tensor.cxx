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
        return storage_->get_data_descriptor();
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
    icdl_proto::Tensor Tensor::serialize() const{
        icdl_proto::Tensor t;
        switch(dtype()){
            case kFloat32: 
                t.set_dtype(icdl_proto::Tensor_TensorDataType::Tensor_TensorDataType_FLOAT_32);
                break;
            case kFloat16:
                t.set_dtype(icdl_proto::Tensor_TensorDataType::Tensor_TensorDataType_FLOAT_16);
                break;
            case kFixpoint:
                t.set_dtype(icdl_proto::Tensor_TensorDataType::Tensor_TensorDataType_FIXPOINT);
                break;
            case TensorDataType::INVALID_DTYPE:
                t.set_dtype(icdl_proto::Tensor_TensorDataType::Tensor_TensorDataType_INVALID_DTYPE);
                break;
            default:
                throw std::runtime_error("Unknown Data Type for serialization");
                break;
        }
        switch(get_mem_layout()){
            case kDense:
                t.set_mem_layout(icdl_proto::Tensor_TensorMemLayout_DENSE_LAYOUT);
                break;
            case kSparse:
                t.set_mem_layout(icdl_proto::Tensor_TensorMemLayout_SPARSE_LAYOUT);
                break;
            case TensorMemLayout::INVALID_LAYOUT:
                t.set_mem_layout(icdl_proto::Tensor_TensorMemLayout_INVALID_LAYOUT);
                break;
            default:
                throw std::runtime_error("Unknowm mem layout for serilization");
                break;
        }
        for(auto dim : size_){
            t.add_tensor_size(dim);
        }
        
        (*t.mutable_storage()) = storage_->serialize();
        return t;
    }

    void Tensor::deserialize(const icdl_proto::Tensor& tensor_proto){
        // sanity check...
        assert(static_cast<size_t>(tensor_proto.tensor_size_size()) == size_.size());
        for(size_t i = 0; i < size_.size(); i++){
            assert(size_[i] == tensor_proto.tensor_size(i));
        }
        TensorDataType proto_data_type;
        TensorMemLayout proto_layout;
        switch(tensor_proto.dtype()){
            case icdl_proto::Tensor::TensorDataType::Tensor_TensorDataType_FLOAT_32: proto_data_type = kFloat32; break;
            case icdl_proto::Tensor::TensorDataType::Tensor_TensorDataType_FLOAT_16: proto_data_type = kFloat16; break;
            case icdl_proto::Tensor::TensorDataType::Tensor_TensorDataType_FIXPOINT: proto_data_type = kFixpoint; break;
            case icdl_proto::Tensor::TensorDataType::Tensor_TensorDataType_INVALID_DTYPE: proto_data_type = TensorDataType::INVALID_DTYPE;break;
            default: {
                throw std::runtime_error("The protobuf's TensorDataType is not compatible with the definition in icdl source.");
            }
        }

        switch(tensor_proto.mem_layout()){
            case icdl_proto::Tensor_TensorMemLayout_DENSE_LAYOUT: proto_layout = kDense; break;
            case icdl_proto::Tensor_TensorMemLayout_INVALID_LAYOUT: proto_layout = TensorMemLayout::INVALID_LAYOUT;break;
            case icdl_proto::Tensor_TensorMemLayout_SPARSE_LAYOUT: proto_layout = kSparse; break;
            default: {
                throw(std::runtime_error("The protobuf's TensorMemLayout is not compatible with the definition in icdl source."));
            }
        }
        assert(proto_data_type == dtype() && proto_layout == mem_layout_);
        storage_->deserialize(tensor_proto.storage());
    }

    Tensor Tensor::convert_to_fixpoint(const StorageConverter& storage_converter,const FixpointRepresent & target_fix_represent,  const TensorMemLayout& target_mem_layout) const{
        Tensor new_tensor;
        StoragePtr new_storage;
        if(this->dtype() == kFixpoint){
            new_storage = storage_converter.fix_to_fix_convert(
                storage_,
                this->storage_->get_data_represent(),//src_fix_represent
                target_fix_represent, //target_fix_represent
                this->get_mem_layout(),//src_mem_layout
                target_mem_layout
            );
        }
        else if(this->dtype() == kFloat32){
            new_storage = storage_converter.float32_to_fix_convert(
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
        assert(new_storage->get_data_type() == kFixpoint);
        new_tensor.mem_layout_ = target_mem_layout;
        new_tensor.size_ = size_;
        assert(new_storage->get_data_represent() == target_fix_represent);
        new_tensor.storage_ = new_storage;

        return new_tensor;
    }

    Tensor Tensor::convert_to_float32(const StorageConverter& storage_converter, const TensorMemLayout& target_mem_layout) const{
        //create a new tensor
        //Tensor new_tensor(*this);
        Tensor new_tensor;
        StoragePtr new_storage;
        if(this->dtype() == kFixpoint){
            new_storage = storage_converter.fix_to_float32_convert(
                storage_,
                this->storage_->get_data_represent(), //src_fix_represent
                this->get_mem_layout(), 
                target_mem_layout
            );
            
        }
        else if(this->dtype() == kFloat32){
            new_storage = storage_converter.float32_to_float32_convert(
                storage_,
                this->get_mem_layout(),
                target_mem_layout
            );
        }
        else{
            std::cerr << "Try to Convert an INVALID Tensor to Floatpoint" << std::endl;
            exit(EXIT_FAILURE);
        }
        new_tensor.storage_ = new_storage;
        new_tensor.mem_layout_ = target_mem_layout;
        new_tensor.size_ = size_;
        assert(new_tensor.storage_->get_data_type() == kFloat32);

        return new_tensor;
    }

    Tensor Tensor::convert_to(const TensorDataDescriptor& descriptor, const TensorMemLayout& target_mem_layout, const StorageConverter& storage_converter) const{
        if(descriptor.get_dtype() == kFloat32){
            return convert_to_float32(
                storage_converter,
                target_mem_layout
            );
        }
        else if(descriptor.get_dtype() == kFixpoint){
            return convert_to_fixpoint(
                storage_converter,
                descriptor.get_represent().fix_point,
                target_mem_layout
            );
        }
        else{
            if(descriptor.get_dtype() == kFloat16){
                std::cerr << "FLOAT16 is not supported now. Dont convert to FLOAT16!" << std::endl;
            }
            std::cerr << "Invalid TensorDataType for conversion!" << std::endl;
            exit(EXIT_FAILURE);
        }
        //assert(storage_->get_data_type() == descriptor.get_dtype());
    }

    Tensor Tensor::convert_to(const TensorDataDescriptor& descriptor) const{
        return convert_to(descriptor, mem_layout_);
    }

    Tensor Tensor::convert_to(const TensorDataDescriptor& descriptor, const TensorMemLayout& target_mem_layout) const{
        return convert_to(descriptor, target_mem_layout, DefaultStorageConverter::get());
    }
    
    size_t Tensor::nelement() const{
        if(size_.size() == 0){
            return 0;
        }
        size_t num_element = 1;
        for(auto& dim : size_){
            num_element *= dim;
        }
        return num_element;
    }

    Tensor::Tensor(const TensorSize& tensor_size, 
                   const TensorDataDescriptor& data_descriptor,
                   const TensorDataLocation& location,
                   const TensorMemLayout& mem_layout,
                   const OptionalTensorInfo optional_info)
        : size_(tensor_size), mem_layout_(mem_layout),  opt_info_(optional_info){
        
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
        : size_(tensor_size),  mem_layout_(mem_layout),  opt_info_(optional_info){
        
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

    Tensor Tensor::view(const TensorSize& tensor_size) const{
        Tensor new_tensor(*this);//default constructor.
        new_tensor.size_ = tensor_size;
        return new_tensor;
    }
}//namespace icdl
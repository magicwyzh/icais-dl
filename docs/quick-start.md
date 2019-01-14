# Compile and Link
This project is made based on CMAKE. 
To compile and use, there are 3 main dependencies
i.e., googletest, protobuf, libtorch
```bash
# build googletest & protobuf from
$ git clone https://github.com/google/googletest
$ git clone --recursive https://github.com/protocolbuffers/protobuf
# get pytorch lib
wget https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-latest.zip
```
Build them with CXXFLAGS=-D_GLIBCXX_USE_CXX11_ABI=0 and install them together in a deps directory.

It may be troublesome. Ask me for the built libs....

Following is the steps to build icdl library.
```bash
# Start building icdl
cd <icdl_src_path>
mkdir deps
# install googletest/protobuf/libtorch in deps
mkdir build # in <icdl_src_path>
cmake .. && make 
# run unit tests
./bin/test_all
```
# Core Idea Description
For HW designer, what they must need to know to add their hardware/simulator implementations in ICDL are two classes, namely icdl::Operator(include/Operator.h), icdl::OperatorImpl(include/OperatorImpl.h) and Tensor(include/Tensor.h);

## Class: Operator 
'Operator' is the virtual base class of a certain tensor operator, such as Conv2d and ReLU, etc. 
The 'Operator' class does not do actual computation, it just contains info about this operator, such as operator name, parameters(weight, bias), etc. And how this operator performs computations on a Tensor depends on one of the member of the 'Operator' class: 
```cpp
using OpImplPtr = std::shared_ptr<OperatorImpl>;
OpImplPtr _impl; // the member defines the actual computation.
```
The Operator class defines the base interface a tensor operator is required, among which the method 
```cpp
TensorList apply(TensorList& inputs);
```
is the most important one, and what does this method do? Very simple:
```cpp
TensorList apply(TensorList& inputs){
    return _impl->apply(this, inputs);
}
```
It passes the pointer of this 'Operator' object and the inputs to the underlying 'OperatorImpl' object, the latter do the actual work. Other interfaces the 'Operator' class provides are something like 
```cpp
class Operator{
public:
    // as its name
    virtual std::string type_name() const;
    void reset_impl(OpImplPtr impl_ptr);
    // get the output tensor size
    virtual std::vector<TensorSize> output_size(const std::vector<TensorSize>& input_sizes) const; 
    virtual TensorSize output_size(const TensorSize& input_size) const = 0;
    // get the compute_complexity, e.g., number of MAC operations.
    virtual int64_t compute_complexity(const TensorSize& input_size) const;
    // let the operator to record execution time.
    bool profile(bool is_profile);
    // get saved tensors/params like weight/bias, etc.
    std::vector<std::pair<std::string, Tensor*>> get_saved_tensors();
    // as its name
    ProfileResults get_profile_results();
protected:
    // the subclass inherit from Operator should register their saved tensors.
    void _register_tensor(const std::string& tensor_name, Tensor* tensor_ptr);
};
```

## Class: OperatorImpl
This class is for users to extending how the computation of a operator is actually performed as described in the 'Operator' class.
```cpp
class OperatorImpl {
private:
public:
    // get operator info and params from the pointer op, but need use 
    // dynamic_pointer_cast<XXXOp>(op)
    virtual TensorList apply(Operator* op, TensorList& inputs) = 0;
    virtual std::string name() const{
        return "Unspecified_OperatorImpl";
    }
    virtual ~OperatorImpl() = default;
};
```
Generally, the user just need to implement the name() and apply()
```cpp
//Conv2dImpl.cxx
TensorList Conv2dImpl::apply(Operator* op, TensorList& inputs){
    // 1. get operator info, such as weight/bias/in_channels/out_channels from the pointer of op.
    // 2. compute based on info from 1 and the inputs of this function, e.g., 
    //    2a. Do MemCpy from CPU memory to Accelerator Memory w.r.t params/inputs
    //    2b. Send control signals to accelerator to let it compute
    //    2c. Do MemCpy from Accelerator Memory to CPU memory w.r.t results.(optional because the Tensor can always lie in Accelerator memory)
    // 3. Construct icdl::Tensor and return a vector of it as results.
}
```

## Class: Tensor
'Tensor' is a class to hold Tensor. It only provides basic tensor info, like size, number of elements, etc. And provides some method like serialize/deserialize. None of the operator like "+"/"-"/"*" is provided because different hardware backend and implementations should do this.
When customize your own 'OperatorImpl', you will manipulate 'Tensor's.

Interfaces quick glimpse:
```cpp
    class Tensor{
    public:
        // return the data type of it, like Float32/Float16/Fixpoint
        TensorDataType dtype() const;
        // return the size
        TensorSize size() const;
        // return where is the data located in, like CPUMemory/AcceleratorMemory, because in
        // some FPGA platform, the FPGA may have its own DDR RAM. (Experimental feature now, 
        // plz assume always use CPUMemory)
        TensorDataLocation get_data_location() const;
        // return how is the data layout in memory, e.g., Dense layout(most common), 
        // Sparse layout(Experimental feature, dont use sparse layout now)
        TensorMemLayout get_mem_layout() const;
        // return a description about the underlying data. e.g., if it is fixpoint, how 
        // many bits? signed? where is the fractional bits? etc. if it is float, also 
        // provides something like mantissa bits/exponential bits, total bits. 
        TensorDataDescriptor get_data_descript() const;
        // number of element, e.g., tensorsize=[3,3], nelement=3*3 = 9
        size_t nelement() const;
        // get the underlying data pointer, the one who actual implement the 
        // computation of an operator should access the data using the pointer 
        // returned by this method
        void* data_ptr() const;
        // not used now, ignore
        void* aux_info_ptr() const;
        // deserialize to a protobuf object
        void deserialize(const icdl_proto::Tensor& tensor_proto);
        // serialize from a protobuf object
        icdl_proto::Tensor serialize() const;
        /* change data type from float-to-fixpoint or fixpoint-to-fixpoint, etc. 
        *   Retrun a new tensor, the underlying storage is changed.
        *  IMPORTANT: NOT Fully test, and the fixpoint representation may vary from
        *  how you implement in your hardware, so it would be better further provide 
        *  the data converter to to do this! It has a default converter, but may not 
        *  enough for you!.
        */
        Tensor convert_to(const TensorDataDescriptor& descriptor) const;

        Tensor convert_to(const TensorDataDescriptor& descriptor, 
                        const TensorMemLayout& target_mem_layout) const;
        // this one pass a storage converter to do the actual conversion
        Tensor convert_to(const TensorDataDescriptor& descriptor, 
                        const TensorMemLayout& target_mem_layout,
                        const StorageConverter& storage_converter) const;
        
        /** Constructors**/
        // a = icdl::Tensor({3,3,3}, icdl::FixpointDescriptor(8, true, 0));
        // a = icdl::Tensor({1,3,32,32}, icdl::Float32Descriptor());
        Tensor(const TensorSize& tensor_size, 
               const TensorDataDescriptor& data_descriptor,
               const TensorDataLocation& location = kCPUMem,
               const TensorMemLayout& mem_layout = kDense, 
               const OptionalTensorInfo optional_info = OptionalTensorInfo());
        // construct a tensor from existing blob, the storage will NOT own the memory.
        // So you have to delete it by yourself! And be sure that the memory is valid
        // when the 'Tensor' class is in use!
        // by default we consider the tensor is in CPU memory and with dense layout
        // a = icdl::Tensor(image.data_ptr(), {1,3,32,32}, icdl::Float32Descriptor());
        Tensor(void * blob_ptr, 
               const TensorSize& tensor_size, 
               const TensorDataDescriptor& data_descriptor,
               const TensorDataLocation& location = TensorDataLocation::CPU_MEMORY,
               const TensorMemLayout& mem_layout = TensorMemLayout::DENSE_LAYOUT,
               const OptionalTensorInfo optional_info = OptionalTensorInfo()
               );
        // copy constructor and operator=: SHALLOW copy!!
        Tensor(const Tensor & other) = default;
        Tensor& operator=(Tensor&& other) = default;
        // return a new Tensor sharing the storage with current Tensor.
        Tensor view(const TensorSize& tensor_size) const;
        bool operator==(const Tensor& rhs) const;
        bool operator!=(const Tensor& rhs) const;
    };

```
# Simple Example of Build New Operator with Customized Backend Implemenatation
## Define your new operator
Use Conv2d as example, it can also be something like residual block.
```cpp
// Conv2dOperator.h
#include "icdl.h"
// build your Options struct to maintain operator info.
struct Conv2dOptions{
    Conv2dOptions(const size_t input_channels, const size_t output_channels, 
                ExpandingArray<2> kernel_size) 
        : input_channels_(input_channels), output_channels_(output_channels), 
            kernel_size_(kernel_size) {}
    /// The number of input features (columns of the input matrix).
    ICDL_ARG(size_t, input_channels);
    /// The number of output features to produce (columns of the output matrix).
    ICDL_ARG(size_t, output_channels);
    /// Whether to learn and add a bias after the linear transformation.
    ICDL_ARG(bool, with_bias) = false;
    ICDL_ARG(ExpandingArray<2>, kernel_size);
    ICDL_ARG(ExpandingArray<2>, stride) = 1;
    ICDL_ARG(ExpandingArray<2>, padding) = 1;
    ICDL_ARG(ExpandingArray<2>, dilation) = 1;
    ICDL_ARG(TensorDataDescriptor, param_descriptor) = Float32Descriptor();
};//Conv2dOptions

// define new operator
namespace icdl{namespace op{
class Conv2d: public Operator{
// The following macros will add things like:
// private: 
//   Tensor _weight;
//   Tensor _bias;
//   Conv2dOptions _options;
// public:
//   const Tensor& get_weight() const{return _weight;}
//   const Tensor& get_bias() const{return _bias;}
//   const Conv2dOptions& get_options() const{ return _options;}
//   virtual std::string type_name() const override{ return "Conv2d";}
    OP_ADD_TENSOR(weight);
    OP_ADD_TENSOR(bias);
    OP_ADD_OPTIONS(Conv2d);
    OP_ADD_COMMON_FUNCTIONS(Conv2d);
public:
    Conv2d(const size_t input_channels, const size_t output_channels,
            const ExpandingArray<2>& kernel_size, 
            const ExpandingArray<2>& stride = 1,
            const ExpandingArray<2>& padding = 1,
            const ExpandingArray<2>& dilation = 1,
            const bool with_bias = false,
            OpImplPtr impl = makeEmptyOperatorImpl(), 
            const TensorDataDescriptor& param_descriptor=Float32Descriptor(),
            const TensorDataLocation& param_location = kCPUMem,
            const TensorMemLayout& param_mem_layout = kDense
    );
    Conv2d(const Conv2dOptions& options, 
            OpImplPtr impl = makeEmptyOperatorImpl(), 
            const TensorDataLocation& param_location = kCPUMem,
            const TensorMemLayout& param_mem_layout = kDense
    );
    virtual TensorSize output_size(const TensorSize& input_size) const override;
}

// define constructors
Conv2d::Conv2d(const Conv2dOptions& options, 
        OpImplPtr impl, 
        const TensorDataLocation& param_location,
        const TensorMemLayout& param_mem_layout
): Operator(impl), _options(options){
    if(options.with_bias()){
        _bias = Tensor({options.output_channels()}, options.param_descriptor(), param_location, param_mem_layout);
        // use _register_tensor so that the bias can be obtained via Operator*
        _register_tensor("bias", &_bias); 
    }
    auto ksize = options.kernel_size();
    TensorSize weight_size{options.output_channels(), options.input_channels(), static_cast<size_t>(ksize->at(0)), static_cast<size_t>(ksize->at(1))};
    _weight = Tensor(weight_size, options.param_descriptor(), param_location, param_mem_layout);
    _register_tensor("weight", &_weight);
}

// just call the former constructor
Conv2d::Conv2d( const size_t input_channels, 
        const size_t output_channels,
        const ExpandingArray<2>& kernel_size, 
        const ExpandingArray<2>& stride,
        const ExpandingArray<2>& padding,
        const ExpandingArray<2>& dilation,
        const bool with_bias,
        OpImplPtr impl, 
        const TensorDataDescriptor& param_descriptor,
        const TensorDataLocation& param_location,
        const TensorMemLayout& param_mem_layout
): Conv2d(Conv2dOptions(input_channels, output_channels, kernel_size).stride(stride).padding(padding).dilation(dilation).with_bias(with_bias), 
    impl, param_location, param_mem_layout){}
}
// Use Macro to define a Factory function to generate a std::shared_ptr<Operator> to a conv2d.
 OP_FACTORY_REGISTER(Conv2d); 
}// namespace icdl::op
```
## Define your own Conv2dImpl
```cpp
//Conv2dPytorchImpl.h
#include "Conv2dOperator.h"
namespace icdl{ namespace op{
class Conv2dPytorchImpl: public OperatorImpl{
    virtual TensorList apply(Operator* op, TensorList& inputs) override;
    virtual std::string name() const override{
        return "Conv2dPytorchImpl";
    }
};
//write your implementation here...
TensorList Conv2dPytorchImpl::apply(Operator* op, TensorList& inputs){
    auto weight = op->get_weight();
    icdl::Tensor bias;
    if(op->get_options().with_bias()){
        bias = op->get_bias();
    }
    auto image = inputs[0];
    auto image_size = image.size();
    // assume use float32
    auto weight_ptr = static_cast<float*>(weight.data_ptr());
    auto bias_ptr = static_cast<float*>(bias.data_ptr());
    // do what you want to do
    // ....
    output = conv_use_pytorch(weight_ptr, bias_ptr, op->get_options().in_channels, op->get_options().out_channels, op->get_options().kernel_size);
    //wrap as a TensorList.
    return {output};
}
}
// factory function
// generate: std::unique_ptr<OperatorImpl> makeLinearPytorchImpl();
OP_IMPL_FACTORY_REGISTER(Conv2dPytorchImpl);

} //namespace icdl::op
```
What you need to do is to make a new OperatorImpl sub-class and define its ``apply" method.
The factory function registration is optional, but for convenience.
## Define and Run your model
```cpp
// main.cxx
#include "icdl.h"
class Model: public icdl::DynamicComputeGraph{
public: 
    Model(){
        // The icdl::Conv2dOpMake is the Factory function generate 
        // in Conv2dOperator.h
        // It will forward all the arguments to the constructor of Conv2d to make an Object and return a std::shared_ptr.
        // The icdl::makeConv2dPytorchImpl is the Factory function generate
        // in Conv2dPytorchImpl.h
        add_compute_node("conv1", icdl::Conv2dOpMake(3,10,3, icdl::makeConv2dPytorchImpl()));
        add_compute_node("conv2", icdl::Conv2dOpMake(10,10,3, icdl::makeConv2dPytorchImpl()));
    };
    virtual TensorList apply(TensorList& inputs) override{
        auto x = inputs[0];
        x = _compute_node["conv1"].apply(x);
        x = _compute_node["conv2"].apply(x);
        return {x};
    }
}

void main(){
    auto model = Model();
    auto image = load_image("xxx.jpg");
    auto image_ptr = image.data_ptr();
    auto input_tensor = icdl::Tensor(image_ptr, {1,3,32,32});
    auto output = model(input_tensor);
    // do what you want to do to the output
    // ...
}
```
To build a model and run, just inherit from icdl::DynamicComputeGraph or icdl::SeqDyComputeGraph. The latter execute each nodes sequentially as the order you add it and dont need to define the ``apply" method.

## Some useful Macros and Data Types
See:

* include/tensor_utils.h. Defines something like TensorDataType, TensorMemLocation, FixpointRepresentation...etc.
* include/operators/pytorch_backend_utils.h For convenience
* include/arg_utils.h : Use MACROs in it to generate OperatorOptions & Factory functions for Operators(Impl).

# Directories and File Description
* docs: this directory. Contain documents.
* include: Head files of core part, like Tensor, Operator, ComputeGraph, etc.
* src: Source files. src directory has the same structure as include, except all files contains actual definition of functions/classes rather than declarations
* test: unit test based on google test.
* include/operators: Contains all actual operators and Impls like Conv2d/Linear, etc.
* include/protos: protobuf related headers.

# Notes
* Float32 should be able to be used easily now. Fixpoint need to further modified to support more functions. 
* Only DenseMemLayout is extensively verified now. Dont use SparseMemLayout.
* Just assume all things in CPUMem, support for AcceleratorMem will be available after i have a FPGA board and test.
# About ICDL (ICAIS DL)
A personal project for fun, but hope can be useful somewhere.

Expected to be a light-weight framework for DL inference accelerator designer. It does not provide actual compute codes, but just provide some interfaces for hardware designers to insert their driver codes for their own accelerator. 

For example, when someone have designed and implemented a Convolution accelerator, then he just need to write codes about how to send control signals like "start", "kernel sizes", "strides", etc, to the accelerator. ICDL provides him an interface to write this part of code. Then he can build an entire neural networks together with other tensor operations like image pre-processing and output post-processing (of course these two parts can also be implemented via special hardware using the same interface in ICDL) via ICDL's neural network building APIs rather than writing an entire application from scratch.

ICDL library is designed to be simple enough that one can easily go across it and make their own extensions, so none of the complicated methods of Tensors is provided here. Tensors are manipulated via various Tensor Operators. Those Tensor Operators can be defined by the user, e.g., the user can define a single Operator equivalent to ``Conv-BN-ReLU-ResidualAdd" rather than use many small operators, so that the user can customize a particular hardware accelerator for it. And this operator can be dynamically bound to different compute backends(OperatorImpl). The users are expected to only provide a specific OperatorImpl.


Things can be customized for users according to their hardware design: 
* Tensor & TensorStorage:
   
   * DataType: Float32 vs. Fixpoint (with fixpoint descriptor)
   * Memory layout: Dense vs. Sparse
   * Location: CPUMemory vs. AcceleratorMemory
   * AcceleratorMemory's Allocator/DeAllocator (like GPU Memory)
* Operator:
   * Operator: Add new operator, e.g., describe conv-BN-Relu as one operator if the accelerator's computational primitive is something like this.
   * OperatorImpls: Depends on how the accelerator is designed

Build a model and use it:
```cpp
#include "icdl.h"
class ResidualBlock: public icdl::DynamicComputeGraph{
public: 
    ResidualBlock(){
        // makeConv2dPytorchImpl is a factory func provides a backend engine for a specific operator. 
        //The user should provide their own implementations. Here we just call it a 'PytorchImpl' which calls pytorch's APIs.
        add_compute_node("conv1", icdl::Conv2dOpMake(3, 10, 3, icdl::op::makeConv2dPytorchImpl()));
        add_compute_node("ReLU1", icdl::ReLUOpMake(icdl::op::makeReluPytorchImpl()));
        add_compute_node("ResAdd", icdl::EltAddOpMake(icdl::op::makeEltAddPytorchImpl()));
    }
    virtual TensorList apply(TensorList& inputs) override{
        auto x = inputs[0];
        auto res = x;
        x = _compute_node["conv1"].apply(x);
        x = _compute_node["ReLU1"].apply(x);
        return _compute_node["ResAdd"].apply({x, res});
    }
}
// inherit from sequential graph, dont need to override apply()
class Model: public icdl::SeqDyComputeGraph{
    Model(){
        add_compute_node({
            {"Conv1", icdl::Conv2dOpMake(3, 10, 3, icdl::op::makeConv2dPytorchImpl())},
            {"Conv2", icdl::Conv2dOpMake(10, 3, 3, icdl::op::makeConv2dPytorchImpl())}
        });
        add_compute_node("ResBlock1", std::make_shared<ResidualBlock>());
    }
}

int main(){
    Model model;
    // fixpoint tensor with 8bit per data, signed, 0 fractional bit.
    auto image = icdl::Tensor({1,3,32,32}, icdl::FixpointTensorDescriptor(8, true, 0)));
    // pre-processing for image...
    // ...
    auto results = model(image);
    // post-processing for results
    // ...
    return 0;
}
```


# TODO
ICDL is in developing and how it will be like in the future is not quite clear now.
Here are something in the to-do list now:
* Provide model load/save APIs that reads/write model params via protobuf.
    * ~~So that we dont need to always play with random/uninitialized params...~~
    * Python script that can transform dict as the GraphParams Protobuf to ICDL-recognizable protobuf file.
* Provide more operators like Conv2d, Activation, etc. The first goal is to provide all operators in Yolo. 
    * NN Ops: Conv2d, ReLU, BatchNorm, Yolo, Concat, ResAdd
    * Utils Op: FixpointTensor Data Conversion, including fix-to-float and vice versa, and fixpoint with different decimal point.
    * Image PreProcessing Ops (and implementations): normalization, etc.
* Dynamic backend APIs:
    * Let the model can change their OperatorImpl, not fixed when model definition. E.g., the first conv of Res50 with 7x7 kernel size is run on cpu while others conv with 1x1&3x3 run on DLA. Backend of each Op should be able to change. The model definition should not show too much info about OperatorImpl. Currently the way is not good.
    * ~~By default the operator constructor should use emptyOperatorImpl.~~
* Graph Params conversion APIs
    * float-fix conversion
    * Memory location conversion
* Discuss with quantization guys about what they want to represent TensorData
    * Currently there is just a FixpointRepresentation with total_bits, frac_bits, sign. However, they may want something like per-channel scalar, etc.
    * New FixpointRepresent implementation and unit test.
* Arm Compute Library Backend
* Scripts that transforms models in framework like Darknet/Pytorch to ICDL model.
    * Not clear now.
* Profiling APIs
    * ~~Profiler RAII class.~~
    * ~~Compute Complexities APIs for Operator and ComputeGraph~~
    * Time/Cycle Logging APIs (can be used by hardware simulator backend)
    * Not quite clear now, but expect that after a graph has run for some times, it can then generates some performance profiling info/forms/timeline, etc. 
    ```cpp
        ResNet50 model();
        model.profile(true);
        output = model(image);
        model.get_profiling_results("res50.icdl_prof");
    ```
* Some pre-defined models (low priority)
    * ResNet-18
    * MobileNet
    * Darknet-53(YoloV3)

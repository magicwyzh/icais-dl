#pragma once 
#include "Operator.h"
#include <string>
#include <array>
#include "arg_utils.h"
namespace icdl{namespace op{

    /**
     * @brief ActivationType
     * Remember to add new string in the definition of ActivationType_to_str
     * when adding new ActivationType!
     * 
     */
    enum class ActivationType: size_t{
        RELU = 0,
        RELU_6 = 1,
        SIGMOID = 2,
        TANH = 3,
        INVALID_ACTIVATION = 4
    };
    static std::array<std::string, 5> act_type_str{
        "RELU",
        "RELU_6",
        "SIGMOID",
        "TANH",
        "INVALID_ACTIVATION"
    };
    const std::string& ActivationType_to_str(const ActivationType& act_type);

    struct ActivationOptions{
        ActivationOptions(const ActivationType& act_type): act_type_(act_type){}
        ICDL_ARG(ActivationType, act_type) = ActivationType::INVALID_ACTIVATION;
    };

    class Activation: public icdl::Operator{
    OP_ADD_OPTIONS(Activation);
    public:
        Activation(const ActivationOptions& act_options, OpImplPtr impl = makeEmptyOperatorImpl())
            : Operator(impl), _options(act_options){}
        virtual TensorSize output_size(const TensorSize& input_size) const override{
            return input_size;
        }
        virtual std::string type_name() const override{
            return ActivationType_to_str(get_options().act_type());
        }
    };
}
    OP_FACTORY_REGISTER(Activation);
}//namespace icdl::op
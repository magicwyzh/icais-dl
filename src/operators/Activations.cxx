#include "operators/Activation.h"

namespace icdl{namespace op{

    const std::string& ActivationType_to_str(const ActivationType& act_type){
        return act_type_str.at(static_cast<size_t>(act_type));
    }

}}
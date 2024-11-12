#ifndef OP_PROTO_H_
#define OP_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(IsClose)
    .INPUT(x1, ge::TensorType::ALL())
    .INPUT(x2, ge::TensorType::ALL())
    .OUTPUT(y, ge::TensorType::ALL())
    .ATTR(rtol, Float, 1e-05)
    .ATTR(atol, Float, 1e-08)
    .ATTR(equal_nan, Bool, false)
    .OP_END_FACTORY_REG(IsClose);

}

#endif

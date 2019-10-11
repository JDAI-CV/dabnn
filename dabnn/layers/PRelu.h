// Copyright 2019 JD.com Inc. JD AI

#ifndef BNN_PRELU_H
#define BNN_PRELU_H

#include <dabnn/layer.h>

namespace bnn {
class PRelu : public Layer {
   public:
    MatCP data_mat;
    MatCP slope_mat;

    PRelu(NetCP net, const std::string &name, css data, css slope)
        : Layer(net, name, "PRelu"), data_mat(mat(data)), slope_mat(mat(slope)) {}
    virtual void forward_impl() const;
};
}  // namespace bnn

#endif /* BNN_PRELU_H */


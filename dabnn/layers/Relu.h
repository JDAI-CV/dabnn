// Copyright 2019 JD.com Inc. JD AI

#ifndef BNN_RELU_H
#define BNN_RELU_H

#include <dabnn/layer.h>

namespace bnn {
class Relu : public Layer {
   public:
    MatCP data_mat;

    Relu(NetCP net, const std::string &name, css data)
        : Layer(net, name, "Relu"), data_mat(mat(data)) {}
    virtual void forward_impl() const;
};
}  // namespace bnn

#endif /* BNN_RELU_H */

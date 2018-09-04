// Copyright 2019 JD.com Inc. JD AI

#ifndef BNN_AFFINE_H
#define BNN_AFFINE_H

#include <dabnn/layer.h>

namespace bnn {
class Affine : public Layer {
   public:
    MatCP data_mat;
    MatCP a_mat;
    MatCP b_mat;

    Affine(NetCP net, const std::string &name, css data, css a, css b)
        : Layer(net, name, "Affine"),
          data_mat(mat(data)),
          a_mat(mat(a)),
          b_mat(mat(b)) {}
    virtual void forward_impl() const;
};
}  // namespace bnn

#endif /* BNN_AFFINE_H */

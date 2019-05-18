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

#ifdef BNN_CHECK_CONSISTENCY
    MatCP output_mat;

    Affine(NetCP net, const std::string &name, css data, css a, css b,
           css output)
        : Layer(net, name, "Affine"),
          data_mat(mat(data)),
          a_mat(mat(a)),
          b_mat(mat(b)),
          output_mat(mat(output)) {}
#else
    Affine(NetCP net, const std::string &name, css data, css a, css b)
        : Layer(net, name, "Affine"),
          data_mat(mat(data)),
          a_mat(mat(a)),
          b_mat(mat(b)) {}
#endif
    virtual void forward_impl() const;
};
}  // namespace bnn

#endif /* BNN_AFFINE_H */

// Copyright 2019 JD.com Inc. JD AI

#ifndef BNN_ADD_H
#define BNN_ADD_H

#include <dabnn/layer.h>

namespace bnn {
class Add : public Layer {
   public:
    MatCP input1_mat;
    MatCP input2_mat;

#ifdef BNN_CHECK_CONSISTENCY
    MatCP output_mat;

    Add(NetCP net, const std::string &name, css input1, css input2, css output)
        : Layer(net, name, "Add"),
          input1_mat(mat(input1)),
          input2_mat(mat(input2)),
          output_mat(mat(output)) {}
#else
    Add(NetCP net, const std::string &name, css input1, css input2)
        : Layer(net, name, "Add"),
          input1_mat(mat(input1)),
          input2_mat(mat(input2)) {}
#endif
    virtual void forward_impl() const;
    ~Add() {}
};
}  // namespace bnn

#endif /* BNN_ADD_H */

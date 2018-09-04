// Copyright 2019 JD.com Inc. JD AI

#ifndef BNN_BINARIZE_H
#define BNN_BINARIZE_H

#include <dabnn/layer.h>

namespace bnn {
class Binarize : public Layer {
   public:
    MatCP input_mat;
    MatCP output_mat;

    Binarize(NetCP net, const std::string &name, css &input, css &output)
        : Layer(net, name, "Binarize"),
          input_mat(mat(input)),
          output_mat(mat(output)) {}
    virtual void forward_impl() const;
};
}  // namespace bnn

#endif /* BNN_BINARIZE_H */

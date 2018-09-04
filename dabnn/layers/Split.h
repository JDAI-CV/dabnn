// Copyright 2019 JD.com Inc. JD AI

#ifndef BNN_SPLIT_H
#define BNN_SPLIT_H

#include <dabnn/layer.h>

namespace bnn {
class Split : public Layer {
   public:
    MatCP input_mat;
    MatCP output_mat1;
    MatCP output_mat2;

    Split(NetCP net, css &name, css &input, css &output1, css &output2)
        : Layer(net, name, "Split"),
          input_mat(mat(input)),
          output_mat1(mat(output1)),
          output_mat2(mat(output2)) {
        BNN_ASSERT(input_mat->data_type == DataType::Bit,
                   "Split only supports bit mat");
        BNN_ASSERT(input_mat->elem_c % 128 == 0 && input_mat->elem_c <= 512,
                   input_mat->elem_c);
    }
    virtual void forward_impl() const;
};
}  // namespace bnn

#endif /* BNN_SPLIT_H */

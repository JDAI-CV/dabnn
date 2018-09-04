// Copyright 2019 JD.com Inc. JD AI

#ifndef BNN_CONCAT_H
#define BNN_CONCAT_H

#include <dabnn/layer.h>

namespace bnn {
class Concat : public Layer {
   public:
    MatCP input1_mat;
    MatCP input2_mat;
    MatCP output_mat;

    Concat(NetCP net, const std::string &name, css input1, css input2,
           css output)
        : Layer(net, name, "Concat"),
          input1_mat(mat(input1)),
          input2_mat(mat(input2)),
          output_mat(mat(output)) {
        BNN_ASSERT(input1_mat->c == input2_mat->c,
                   "channels of two inputs should equal");
        BNN_ASSERT(input1_mat->data_type == DataType::Float,
                   "input1 data type should be float");
        BNN_ASSERT(input2_mat->data_type == DataType::Float,
                   "input2 data type should be float");
    }
    virtual void forward_impl() const;
};
}  // namespace bnn

#endif /* BNN_CONCAT_H */

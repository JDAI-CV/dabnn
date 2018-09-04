// Copyright 2019 JD.com Inc. JD AI

#ifndef BNN_SHUFFLE_H
#define BNN_SHUFFLE_H

#include <dabnn/layer.h>

namespace bnn {
class Shuffle : public Layer {
   public:
    MatCP data_mat;

    Shuffle(NetCP net, css &name, css &data)
        : Layer(net, name, "Shuffle"), data_mat(mat(data)) {
        BNN_ASSERT(data_mat->data_type == DataType::Bit,
                   "Shuffle only supports bit mat");
        BNN_ASSERT(data_mat->elem_c % 128 == 0 && data_mat->elem_c <= 512,
                   data_mat->elem_c);
    }
    virtual void forward_impl() const;
};
}  // namespace bnn

#endif /* BNN_SHUFFLE_H */

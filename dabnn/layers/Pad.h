// Copyright 2019 JD.com Inc. JD AI

#ifndef BNN_PAD_H
#define BNN_PAD_H

#include <dabnn/layer.h>

namespace bnn {
class Pad : public Layer {
   public:
    MatCP input_mat;
    MatCP output_mat;
    int pad_h;
    int pad_w;
    float val;

    Pad(NetCP net, const std::string &name, css &input, int pad_h, int pad_w,
        float val, css &output)
        : Layer(net, name, "Pad"),
          input_mat(mat(input)),
          output_mat(mat(output)),
          pad_h(pad_h),
          pad_w(pad_w),
          val(val) {}
    virtual void forward_impl() const;
};
}  // namespace bnn

#endif /* BNN_PAD_H */

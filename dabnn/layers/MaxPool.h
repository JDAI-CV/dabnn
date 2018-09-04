// Copyright 2019 JD.com Inc. JD AI

#ifndef BNN_MAXPOOL_H
#define BNN_MAXPOOL_H

#include <dabnn/layer.h>

namespace bnn {
class MaxPool : public Layer {
   public:
    MatCP input_mat;
    std::shared_ptr<Mat> padded_mat;
    MatCP output_mat;
    int kernel_h;
    int kernel_w;
    int pad_h;
    int pad_w;
    int stride_h;
    int stride_w;

    MaxPool(NetCP net, const std::string &name, css input, css output,
            int kernel_h, int kernel_w, int pad_h, int pad_w, int stride_h,
            int stride_w);
    virtual void forward_impl() const;
    virtual std::string to_str() const;
};
}  // namespace bnn

#endif /* BNN_MAXPOOL_H */

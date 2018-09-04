// Copyright 2019 JD.com Inc. JD AI

#ifndef BNN_FLOATCONV_H
#define BNN_FLOATCONV_H

#include <dabnn/layer.h>

namespace bnn {
class FloatConv : public Layer {
   public:
    MatCP input_mat;
    MatCP weight_mat;
    MatCP bias_mat;
    MatCP output_mat;
    const int pad_h;
    const int pad_w;
    const int stride_h;
    const int stride_w;
    const int dilation;

    FloatConv(NetCP net, const std::string &name, css input, css weight,
              css output, int pad_h, int pad_w, int stride_h, int stride_w,
              int dilation)
        : Layer(net, name, "Float Conv"),
          input_mat(mat(input)),
          weight_mat(mat(weight)),
          output_mat(mat(output)),
          pad_h(pad_h),
          pad_w(pad_w),
          stride_h(stride_h),
          stride_w(stride_w),
          dilation(dilation) {}

    FloatConv(NetCP net, const std::string &name, css input, css weight,
              css bias, css output, int pad_h, int pad_w, int stride_h,
              int stride_w, int dilation)
        : Layer(net, name, "Float Conv"),
          input_mat(mat(input)),
          weight_mat(mat(weight)),
          bias_mat(mat(bias)),
          output_mat(mat(output)),
          pad_h(pad_h),
          pad_w(pad_w),
          stride_h(stride_h),
          stride_w(stride_w),
          dilation(dilation) {}

    virtual void forward_impl() const;
    virtual std::string to_str() const;
};
}  // namespace bnn

#endif /* BNN_FLOATCONV_H */

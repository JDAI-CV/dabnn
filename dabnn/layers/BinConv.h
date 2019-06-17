// Copyright 2019 JD.com Inc. JD AI

#ifndef BNN_BINCONV_H
#define BNN_BINCONV_H

#include <dabnn/layer.h>

namespace bnn {
class BinConv : public Layer {
   public:
    MatCP input_mat;
    MatP binarized_mat;
    MatP padded_mat;
    MatP col_mat;
    MatCP weight_mat;
    MatP transposed_weight_mat;
    MatCP output_mat;
    const int pad_h;
    const int pad_w;
    const int stride_h;
    const int stride_w;

    BinConv(NetCP net, const std::string &name, css input, css weight,
            css output, int pad_h, int pad_w, int stride_h, int stride_w);
    virtual void forward_impl() const;
    virtual std::string to_str() const;

   private:
    bool direct_conv_compatible() const;
    bool gemm_compatible() const;
};
}  // namespace bnn

#endif /* BNN_BINCONV_H */

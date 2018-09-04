// Copyright 2019 JD.com Inc. JD AI

#include "FloatConv.h"

#include <dabnn/fconv.h>

namespace bnn {

void FloatConv::forward_impl() const {
    if (bias_mat == nullptr) {
        fconv(*input_mat, *weight_mat, weight_mat->h, weight_mat->w, pad_h,
              pad_w, stride_h, stride_w, dilation, dilation, output_mat->c,
              *output_mat);
    } else {
        fconv(*input_mat, *weight_mat, *bias_mat, weight_mat->h, weight_mat->w,
              pad_h, pad_w, stride_h, stride_w, dilation, dilation,
              output_mat->c, *output_mat);
    }
}

std::string FloatConv::to_str() const {
    std::stringstream ss;
    ss << "input_h: " << std::to_string(input_mat->h)
       << ", input_w: " << std::to_string(input_mat->w)
       << ", input_c: " << std::to_string(input_mat->c)
       << ", weight_h: " << std::to_string(weight_mat->h)
       << ", weight_w: " << std::to_string(weight_mat->w)
       << ", weight_n: " << std::to_string(weight_mat->n);

    return ss.str();
}

}  // namespace bnn

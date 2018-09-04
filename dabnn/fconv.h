// Copyright 2019 JD.com Inc. JD AI

#ifndef BNN_FCONV_HPP
#define BNN_FCONV_HPP

#include <Eigen/Dense>

#include <common/helper.h>
#include "glog/logging.h"
#include "im2col.h"
#include "mat.h"

namespace bnn {
void fconv(const Mat &input, const Mat &weight, const int kernel_h,
           const int kernel_w, const int pad_h, const int pad_w,
           const int stride_h, const int stride_w, const int dilation_h,
           const int dilation_w, const int output_channels, Mat &output) {
    using namespace Eigen;
    const int output_h =
        (input.h + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h +
        1;
    const int output_w =
        (input.w + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w +
        1;
    const int a = output_h * output_w * kernel_h * kernel_w * input.c;
    Mat input_col(a, input.data_type);

    VLOG(5) << "im2col";
    im2col(input, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w,
           dilation_h, dilation_w, input_col);
    VLOG(5) << "im2col end";
    const int M = output_channels;
    const int N = output_h * output_w;
    const int K = kernel_h * kernel_w * input.c;
    Map<Matrix<float, Dynamic, Dynamic, RowMajor>> weight_eg(
        static_cast<float *>(weight.data), M, K);
    Map<Matrix<float, Dynamic, Dynamic, RowMajor>> input_notrans_eg(
        static_cast<float *>(input_col.data), N, K);
    Map<Matrix<float, Dynamic, Dynamic, ColMajor>> output_eg(
        static_cast<float *>(output.data), M, N);
    output_eg.noalias() = weight_eg * input_notrans_eg.transpose();
}

void fconv(const Mat &input, const Mat &weight, const Mat &bias,
           const int kernel_h, const int kernel_w, const int pad_h,
           const int pad_w, const int stride_h, const int stride_w,
           const int dilation_h, const int dilation_w,
           const int output_channels, Mat &output) {
    using namespace Eigen;
    const int output_h =
        (input.h + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h +
        1;
    const int output_w =
        (input.w + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w +
        1;
    const int a = output_h * output_w * kernel_h * kernel_w * input.c;
    Mat input_col(a, input.data_type);

    VLOG(5) << "im2col";
    im2col(input, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w,
           dilation_h, dilation_w, input_col);
    VLOG(5) << "im2col end";
    // PNT(input_col);
    const int M = output_channels;
    const int N = output_h * output_w;
    const int K = kernel_h * kernel_w * input.c;
    Map<Matrix<float, Dynamic, Dynamic, RowMajor>> weight_eg(
        static_cast<float *>(weight.data), M, K);
    Map<VectorXf> bias_eg(static_cast<float *>(bias.data), M);
    Map<Matrix<float, Dynamic, Dynamic, RowMajor>> input_notrans_eg(
        static_cast<float *>(input_col.data), N, K);
    Map<Matrix<float, Dynamic, Dynamic, ColMajor>> output_eg(
        static_cast<float *>(output.data), M, N);
    output_eg.noalias() = weight_eg * input_notrans_eg.transpose();
    output_eg.colwise() += bias_eg;
}
}  // namespace bnn

#endif /* BNN_FCONV_HPP */

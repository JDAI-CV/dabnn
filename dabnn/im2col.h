// Copyright 2019 JD.com Inc. JD AI

#ifndef BNN_IM2COL_HPP
#define BNN_IM2COL_HPP

#include <cstring>

#include <common/helper.h>
#include "mat.h"

namespace bnn {

// Modified from caffe
inline void im2col(const Mat &im, const int kernel_h, const int kernel_w,
                   const int pad_h, const int pad_w, const int stride_h,
                   const int stride_w, const int dilation_h,
                   const int dilation_w, Mat &col) {
    const int output_h =
        (im.h + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    const int output_w =
        (im.w + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

    char *data_col = static_cast<char *>(col);
    int input_y = 0;
    FORZ(output_y, output_h) {
        int input_x = 0;
        FORZ(output_x, output_w) {
            FORZ(kh, kernel_h) {
                int y = input_y - pad_h + kh * dilation_h;
                const char *data_im = static_cast<char *>(im.data) +
                                      y * im.w * im.c * im.elemsize;
                FORZ(kw, kernel_w) {
                    int x = input_x - pad_w + kw * dilation_w;
                    if (y < 0 || y >= im.h || x < 0 || x >= im.w) {
                        memset(data_col, 0, im.c * im.elemsize);
                    } else {
                        memcpy(data_col, data_im + x * im.c * im.elemsize,
                               im.c * im.elemsize);
                    }
                    data_col += im.c * im.elemsize;
                }
            }
            input_x += stride_w;
        }
        input_y += stride_h;
    }
}

}  // namespace bnn

#endif /* BNN_IM2COL_HPP */

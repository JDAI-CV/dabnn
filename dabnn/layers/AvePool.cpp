// Copyright 2019 JD.com Inc. JD AI

#include "AvePool.h"

#include <dabnn/net.h>
#include <dabnn/pad.h>

namespace bnn {

void ave_pool_2x2_s2(const bnn::Mat &input, bnn::Mat &output) {
    FORZ(h, output.h) {
        FORZ(w, output.w) {
            const float *ptr0 = input.point<float>(h * 2 + 0, w * 2 + 0);
            const float *ptr1 = input.point<float>(h * 2 + 0, w * 2 + 1);
            const float *ptr2 = input.point<float>(h * 2 + 1, w * 2 + 0);
            const float *ptr3 = input.point<float>(h * 2 + 1, w * 2 + 1);
            float *output_ptr = output.point<float>(h, w);
            size_t nn = input.c >> 2;
            asm volatile(
                "fmov   s30, #4.0               \n"
                "dup    v30.4s, v30.s[0]        \n"
                "0:     \n"
                "ld1    {v0.4s}, [%0], #16      \n"
                "prfm   pldl1keep, [%0, #128]   \n"
                "ld1    {v1.4s}, [%1], #16      \n"
                "prfm   pldl1keep, [%1, #128]   \n"
                "ld1    {v2.4s}, [%2], #16      \n"
                "prfm   pldl1keep, [%2, #128]   \n"
                "ld1    {v3.4s}, [%3], #16      \n"
                "prfm   pldl1keep, [%3, #128]   \n"
                "fadd   v0.4s, v0.4s, v1.4s     \n"
                "fadd   v2.4s, v2.4s, v3.4s     \n"
                "fadd   v0.4s, v0.4s, v2.4s     \n"
                "fdiv   v0.4s, v0.4s, v30.4s  \n"
                "subs   %5, %5, #1            \n"
                "st1    {v0.4s}, [%4], #16      \n"
                "bne    0b                      \n"

                : "+r"(ptr0),        // %0
                  "+r"(ptr1),        // %1
                  "+r"(ptr2),        // %2
                  "+r"(ptr3),        // %3
                  "+r"(output_ptr),  // %4
                  "+r"(nn)           // %5
                :
                : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6",
                  "v7", "v8", "v9", "v10", "v11", "v12", "v30");
        }
    }
}

void ave_pool_fallback(const bnn::Mat &input, const size_t pad_h,
                       const size_t pad_w, const size_t stride_h,
                       const size_t stride_w, const size_t kernel_h,
                       const size_t kernel_w, bnn::Mat &output) {
    const int output_h =
        (input.h + 2 * pad_h - ((kernel_h - 1) + 1)) / stride_h + 1;
    const int output_w =
        (input.w + 2 * pad_w - ((kernel_w - 1) + 1)) / stride_w + 1;

    BNN_ASSERT(input.w * input.c * input.elemsize % 16 == 0, "Not align");
    BNN_ASSERT(output.w * output.c * output.elemsize % 16 == 0, "Not align");

    int input_y = 0;
    FORZ(output_y, output_h) {
        int input_x = 0;
        FORZ(output_x, output_w) {
            FORZ(output_c, input.c) {
                size_t n = 0;
                float sum = 0;
                FORZ(kh, kernel_h) {
                    int y = input_y - pad_h + kh;
                    const float *input_ptr = input.point<float>(y, 0);
                    FORZ(kw, kernel_w) {
                        int x = input_x - pad_w + kw;
                        if (!(y < 0 || y >= input.h || x < 0 || x >= input.w)) {
                            const auto val = input_ptr[x * input.c + output_c];
                            sum += val;
                            n++;
                        }
                    }
                }

                output[output_y * output_w * input.c + output_x * input.c +
                       output_c] = sum / n;
            }
            input_x += stride_w;
        }
        input_y += stride_h;
    }
}

AvePool::AvePool(NetCP net, const std::string &name, css input, css output,
                 int kernel_h, int kernel_w, int pad_h, int pad_w, int stride_h,
                 int stride_w)
    : Layer(net, name, "AvePool"),
      input_mat(mat(input)),
      output_mat(mat(output)),
      kernel_h(kernel_h),
      kernel_w(kernel_w),
      pad_h(pad_h),
      pad_w(pad_w),
      stride_h(stride_h),
      stride_w(stride_w) {
    auto &mat_map = net.lock()->mat_map_;
    const auto &pad_name = "pad_for_" + output + "_cal";
    if (mat_map.find(pad_name) == mat_map.end()) {
        auto &input_mat = *mat_map[input];
        mat_map[pad_name] = std::make_shared<Mat>(
            input_mat.h + pad_h * 2, input_mat.w + pad_w * 2, input_mat.c,
            input_mat.data_type, pad_name);
    }
    padded_mat = mat_map[pad_name];
}

void AvePool::forward_impl() const {
    if (stride_h == 2 && stride_w == 2 && kernel_h == 2 && kernel_w == 2 &&
        input_mat->c % 4 == 0) {
        pad(*input_mat, pad_h, pad_w, *padded_mat);
        ave_pool_2x2_s2(*padded_mat, *output_mat);
    } else {
        ave_pool_fallback(*input_mat, pad_h, pad_w, stride_h, stride_w,
                          kernel_h, kernel_w, *output_mat);
    }
}

}  // namespace bnn

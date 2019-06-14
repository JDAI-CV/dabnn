// Copyright 2019 JD.com Inc. JD AI

#include "MaxPool.h"

#include <limits>

#include <dabnn/net.h>
#include <dabnn/pad.h>

namespace bnn {

#ifdef __ARM_NEON
void maxpool2x2(const bnn::Mat &input, bnn::Mat &output, const int stride_h = 1,
                const int stride_w = 1) {
    FORZ(h, output.h) {
        FORZ(w, output.w) {
            const float *ptr0 =
                input.point<float>(h * stride_h + 0, w * stride_w + 0);
            const float *ptr1 =
                input.point<float>(h * stride_h + 0, w * stride_w + 1);
            const float *ptr2 =
                input.point<float>(h * stride_h + 1, w * stride_w + 0);
            const float *ptr3 =
                input.point<float>(h * stride_h + 1, w * stride_w + 1);
            float *output_ptr = output.point<float>(h, w);
            size_t nn = input.c >> 2;
#ifdef __aarch64__
            asm volatile(
                "0:     \n"
                "ld1    {v0.4s}, [%0], #16      \n"
                "prfm   pldl1keep, [%0, #128]   \n"
                "ld1    {v1.4s}, [%1], #16      \n"
                "prfm   pldl1keep, [%1, #128]   \n"
                "ld1    {v2.4s}, [%2], #16      \n"
                "prfm   pldl1keep, [%2, #128]   \n"
                "ld1    {v3.4s}, [%3], #16      \n"
                "prfm   pldl1keep, [%3, #128]   \n"
                "fmax   v0.4s, v0.4s, v1.4s     \n"
                "fmax   v2.4s, v2.4s, v3.4s     \n"
                "fmax   v0.4s, v0.4s, v2.4s     \n"
                "subs   %5, %5, #1              \n"
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
                  "v7", "v8", "v9", "v10", "v11", "v12");
#else   // __aarch64__
            asm volatile(
                "0:     \n"
                "vld1.32    q0, [%0]!       \n"
                "pld    [%0, #128]          \n"
                "vld1.32    q1, [%1]!       \n"
                "pld    [%1, #128]          \n"
                "vld1.32    q2, [%2]!       \n"
                "pld    [%2, #128]          \n"
                "vld1.32    q3, [%3]!       \n"
                "pld    [%3, #128]          \n"
                "vmax.f32   q0, q0, q1      \n"
                "vmax.f32   q2, q2, q3      \n"
                "vmax.f32   q0, q0, q2      \n"
                "subs   %5, %5, #1          \n"
                "vst1.32    q0, [%4]!       \n"
                "bne    0b                  \n"

                : "+r"(ptr0),        // %0
                  "+r"(ptr1),        // %1
                  "+r"(ptr2),        // %2
                  "+r"(ptr3),        // %3
                  "+r"(output_ptr),  // %4
                  "+r"(nn)           // %5
                :
                : "cc", "memory", "q0", "q1", "q2", "q3");
#endif  // __aarch64__
        }
    }
}

void maxpool3x3(const bnn::Mat &input, bnn::Mat &output, const int stride_h = 1,
                const int stride_w = 1) {
    FORZ(h, output.h) {
        FORZ(w, output.w) {
            const float *ptr0 =
                input.point<float>(h * stride_h + 0, w * stride_w + 0);
            const float *ptr1 =
                input.point<float>(h * stride_h + 0, w * stride_w + 1);
            const float *ptr2 =
                input.point<float>(h * stride_h + 0, w * stride_w + 2);
            const float *ptr3 =
                input.point<float>(h * stride_h + 1, w * stride_w + 0);
            const float *ptr4 =
                input.point<float>(h * stride_h + 1, w * stride_w + 1);
            const float *ptr5 =
                input.point<float>(h * stride_h + 1, w * stride_w + 2);
            const float *ptr6 =
                input.point<float>(h * stride_h + 2, w * stride_w + 0);
            const float *ptr7 =
                input.point<float>(h * stride_h + 2, w * stride_w + 1);
            const float *ptr8 =
                input.point<float>(h * stride_h + 2, w * stride_w + 2);
            float *output_ptr = output.point<float>(h, w);
            size_t nn = input.c >> 2;
#ifdef __aarch64__
            asm volatile(
                "0:     \n"
                "ld1    {v0.4s}, [%0], #16      \n"
                "prfm   pldl1keep, [%0, #128]   \n"
                "ld1    {v1.4s}, [%1], #16      \n"
                "prfm   pldl1keep, [%1, #128]   \n"
                "ld1    {v2.4s}, [%2], #16      \n"
                "prfm   pldl1keep, [%2, #128]   \n"
                "ld1    {v3.4s}, [%3], #16      \n"
                "prfm   pldl1keep, [%3, #128]   \n"
                "fmax   v0.4s, v0.4s, v1.4s     \n"
                "ld1    {v4.4s}, [%4], #16      \n"
                "prfm   pldl1keep, [%4, #128]   \n"
                "fmax   v2.4s, v2.4s, v3.4s     \n"
                "ld1    {v5.4s}, [%5], #16      \n"
                "prfm   pldl1keep, [%5, #128]   \n"
                "ld1    {v6.4s}, [%6], #16      \n"
                "prfm   pldl1keep, [%6, #128]   \n"
                "fmax   v4.4s, v4.4s, v5.4s     \n"
                "ld1    {v7.4s}, [%7], #16      \n"
                "prfm   pldl1keep, [%7, #128]   \n"
                "ld1    {v8.4s}, [%8], #16      \n"
                "prfm   pldl1keep, [%8, #128]   \n"
                "fmax   v2.4s, v2.4s, v8.4s     \n"
                "fmax   v6.4s, v6.4s, v7.4s     \n"
                "fmax   v0.4s, v0.4s, v2.4s     \n"
                "fmax   v4.4s, v4.4s, v6.4s     \n"
                "subs   %10, %10, #1              \n"
                "fmax   v0.4s, v0.4s, v4.4s     \n"
                "st1    {v0.4s}, [%9], #16      \n"
                "bne    0b                      \n"

                : "+r"(ptr0),        // %0
                  "+r"(ptr1),        // %1
                  "+r"(ptr2),        // %2
                  "+r"(ptr3),        // %3
                  "+r"(ptr4),        // %4
                  "+r"(ptr5),        // %5
                  "+r"(ptr6),        // %6
                  "+r"(ptr7),        // %7
                  "+r"(ptr8),        // %8
                  "+r"(output_ptr),  // %9
                  "+r"(nn)           // %10
                :
                : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6",
                  "v7", "v8", "v9", "v10", "v11", "v12");
#else
            asm volatile(
                "0:     \n"
                "vld1.32    q0, [%0]!       \n"
                "pld    [%0, #128]          \n"
                "vld1.32    q1, [%1]!       \n"
                "pld    [%1, #128]          \n"
                "vld1.32    q2, [%2]!       \n"
                "pld    [%2, #128]          \n"
                "vld1.32    q3, [%3]!       \n"
                "pld    [%3, #128]          \n"
                "vmax.f32   q0, q0, q1      \n"
                "vld1.32    q4, [%4]!       \n"
                "pld    [%4, #128]          \n"
                "vmax.f32   q2, q2, q3      \n"
                "vld1.32    q5, [%5]!       \n"
                "pld    [%5, #128]          \n"
                "vld1.32    q6, [%6]!       \n"
                "pld    [%6, #128]          \n"
                "vmax.f32   q4, q4, q5      \n"
                "vld1.32    q7, [%7]!       \n"
                "pld    [%7, #128]          \n"
                "vld1.32    q8, [%8]!       \n"
                "pld    [%8, #128]          \n"
                "vmax.f32   q2, q2, q8      \n"
                "vmax.f32   q6, q6, q7      \n"
                "vmax.f32   q0, q0, q2      \n"
                "subs       %10, %10, #1    \n"
                "vmax.f32   q4, q4, q6      \n"
                "vmax.f32   q0, q0, q4      \n"
                "vst1.32    q0, [%9]!       \n"
                "bne    0b                  \n"

                : "+r"(ptr0),        // %0
                  "+r"(ptr1),        // %1
                  "+r"(ptr2),        // %2
                  "+r"(ptr3),        // %3
                  "+r"(ptr4),        // %4
                  "+r"(ptr5),        // %5
                  "+r"(ptr6),        // %6
                  "+r"(ptr7),        // %7
                  "+r"(ptr8),        // %8
                  "+r"(output_ptr),  // %9
                  "+r"(nn)           // %10
                :
                : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6",
                  "q7", "q8");
#endif
        }
    }
}
#endif  // __ARM_NEON

void max_pool_fallback(const bnn::Mat &input, const size_t pad_h,
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
                float m = -std::numeric_limits<float>::max();
                FORZ(kh, kernel_h) {
                    int y = input_y - pad_h + kh;
                    const float *input_ptr = input.point<float>(y, 0);
                    FORZ(kw, kernel_w) {
                        int x = input_x - pad_w + kw;
                        if (!(y < 0 || y >= input.h || x < 0 || x >= input.w)) {
                            const auto val = input_ptr[x * input.c + output_c];
                            m = std::max(m, val);
                        }
                    }
                }

                output[output_y * output_w * input.c + output_x * input.c +
                       output_c] = m;
            }
            input_x += stride_w;
        }
        input_y += stride_h;
    }
}

MaxPool::MaxPool(NetCP net, const std::string &name, css input, css output,
                 int kernel_h, int kernel_w, int pad_h, int pad_w, int stride_h,
                 int stride_w)
    : Layer(net, name, "MaxPool"),
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
void MaxPool::forward_impl() const {
#ifdef __ARM_NEON
    if (kernel_h == 3 && kernel_w == 3) {
        // std::numeric_limits<float>::min() is the closest value to 0, so we
        // uses -max()
        pad(*input_mat, pad_h, pad_w, *padded_mat,
            -std::numeric_limits<float>::max());
        maxpool3x3(*padded_mat, *output_mat, stride_h, stride_w);
    } else if (kernel_h == 2 && kernel_w == 2) {
        pad(*input_mat, pad_h, pad_w, *padded_mat,
            -std::numeric_limits<float>::max());
        maxpool2x2(*padded_mat, *output_mat, stride_h, stride_w);
    } else {
        max_pool_fallback(*input_mat, pad_h, pad_w, stride_h, stride_w,
                          kernel_h, kernel_w, *output_mat);
    }
#else
    max_pool_fallback(*input_mat, pad_h, pad_w, stride_h, stride_w, kernel_h,
                      kernel_w, *output_mat);
#endif  // __aarch64__
}

std::string MaxPool::to_str() const {
    std::stringstream ss;
    ss << type_ << ", ";
    PNT_TO(ss, kernel_h, kernel_w, stride_h, stride_w);
    return ss.str();
}

}  // namespace bnn

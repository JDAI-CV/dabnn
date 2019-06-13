// Copyright 2019 JD.com Inc. JD AI

#ifndef BCONV_H
#define BCONV_H

#if __ARM_NEON
#include <arm_neon.h>
#endif  // __ARM_NEON

#if not defined (__aarch64__)
#include <common/baseline.h>
#endif
#include <common/helper.h>
#include <dabnn/im2col.h>
#include "mat.h"

namespace bnn {
#ifdef __aarch64__
inline void bconv_1x1_64(const Mat &bottom_blob, const Mat &weight,
                         Mat &top_blob);
inline void bconv_1x1_128(const Mat &bottom_blob, const Mat &weight,
                          Mat &top_blob);
inline void bconv_1x1_256(const Mat &bottom_blob, const Mat &weight,
                          Mat &top_blob);
inline void bconv_1x1_512(const Mat &bottom_blob, const Mat &weight,
                          Mat &top_blob);
#endif
inline void bconv_3x3(const Mat &bottom_blob, const Mat &weight, Mat &top_blob,
                      const int stride = 1);
#ifdef __aarch64__
inline void bconv_3x3_64(const Mat &bottom_blob, const Mat &weight,
                         Mat &top_blob, const int stride = 1);
inline void bconv_3x3_64_fallback(const Mat &bottom_blob, const Mat &weight,
                                  Mat &top_blob, const int stride = 1);
inline void bconv_3x3_64_opt(const Mat &bottom_blob, const Mat &weight,
                             Mat &top_blob);
inline void bconv_3x3_64_opt2(const Mat &bottom_blob, const Mat &weight,
                              Mat &top_blob, const int pad = 0,
                              const int stride = 1);
inline void bconv_3x3_64_opt3(const Mat &bottom_blob, const Mat &weight,
                              Mat &top_blob, const int pad = 0,
                              const int stride = 1);
inline void bconv_3x3_64_opt4(const Mat &bottom_blob, const Mat &weight,
                              Mat &top_blob, const int pad = 0,
                              const int stride = 1);
inline void bconv_3x3_128_internal_s1(const uint64_t *bottom_ptr, const int b_w,
                                      const uint64_t *weight_ptr,
                                      float *top_ptr, const int top_h,
                                      const int top_w);
inline void bconv_3x3_128_internal_fallback(
    const uint64_t *bottom_ptr, const int b_w, const uint64_t *weight_ptr,
    float *top_ptr, const int top_h, const int top_w, const int stride = 1);
#endif
}  // namespace bnn

#ifdef __aarch64__
inline void bnn::bconv_3x3_64(const Mat &bottom_blob, const Mat &weight,
                              Mat &top_blob, const int stride) {
    bconv_3x3_64_opt4(bottom_blob, weight, top_blob, 0, stride);
}

inline void bnn::bconv_3x3_64_opt3(const Mat &bottom_blob, const Mat &weight,
                                   Mat &top_blob, const int pad,
                                   const int stride) {
    /**
     * See bconv_3x3_64_opt4
     */
    static uint64_t col_buf[999999];

    const size_t col_h = weight.h * weight.w;
    const size_t col_w = top_blob.h * top_blob.w * top_blob.c;
    const size_t col_len = col_h * col_w;

    Mat col(col_len, col_buf, DataType::Bit);

    im2col(bottom_blob, 3, 3, pad, pad, stride, stride, 1, 1, col);

    uint64_t *col_ptr = col_buf;
    uint64_t *weight_ptr = static_cast<uint64_t *>(weight.data);
    float *output_ptr = static_cast<float *>(top_blob.data);

    size_t nn1 = top_blob.c / 4;
    size_t nn2 = top_blob.h * top_blob.w / 4;
    /*
     * v0~v3 contain input, v4~v7 contain weight
     * v8~v23 contain output
     * v24~v30 contain temporary values
     */
    asm volatile(
        "2:     \n"
        "mov    x1, %1      \n"
        "mov    x0, %3      \n"
        "1:     \n"
        "mov    x2, #5      \n"
        "mov    x3, %0      \n"
        "0:     \n"

        "eor    v24.16b, v0.16b, v4.16b     \n"
        "eor    v25.16b, v0.16b, v5.16b     \n"
        "eor    v26.16b, v0.16b, v6.16b     \n"
        "eor    v27.16b, v0.16b, v7.16b     \n"
        "eor    v28.16b, v1.16b, v4.16b     \n"
        "eor    v29.16b, v1.16b, v5.16b     \n"
        "cnt    v24.16b, v24.16b            \n"
        "cnt    v25.16b, v25.16b            \n"
        "cnt    v26.16b, v26.16b            \n"
        "cnt    v27.16b, v27.16b            \n"
        "cnt    v28.16b, v28.16b            \n"
        "cnt    v29.16b, v29.16b            \n"
        "add    v8.16b, v8.16b, v24.16b     \n"
        "add    v9.16b, v9.16b, v25.16b     \n"
        "add    v10.16b, v10.16b, v26.16b     \n"
        "add    v11.16b, v11.16b, v27.16b     \n"
        "add    v12.16b, v12.16b, v28.16b     \n"
        "add    v13.16b, v13.16b, v28.16b     \n"
        "eor    v24.16b, v1.16b, v6.16b     \n"
        "eor    v25.16b, v1.16b, v7.16b     \n"
        "eor    v26.16b, v2.16b, v4.16b     \n"
        "eor    v27.16b, v2.16b, v5.16b     \n"
        "eor    v28.16b, v2.16b, v6.16b     \n"
        "eor    v29.16b, v2.16b, v7.16b     \n"
        "cnt    v24.16b, v24.16b            \n"
        "cnt    v25.16b, v25.16b            \n"
        "cnt    v26.16b, v26.16b            \n"
        "cnt    v27.16b, v27.16b            \n"
        "cnt    v28.16b, v28.16b            \n"
        "cnt    v29.16b, v29.16b            \n"
        "add    v14.16b, v14.16b, v24.16b     \n"
        "add    v15.16b, v15.16b, v25.16b     \n"
        "add    v16.16b, v16.16b, v26.16b     \n"
        "add    v17.16b, v17.16b, v27.16b     \n"
        "add    v18.16b, v18.16b, v28.16b     \n"
        "add    v19.16b, v19.16b, v29.16b     \n"
        "eor    v24.16b, v3.16b, v4.16b     \n"
        "eor    v25.16b, v3.16b, v5.16b     \n"
        "eor    v26.16b, v3.16b, v6.16b     \n"
        "eor    v27.16b, v3.16b, v7.16b     \n"
        "subs   x2, x2, #1  \n"
        "cnt    v24.16b, v24.16b            \n"
        "cnt    v25.16b, v25.16b            \n"
        "cnt    v26.16b, v26.16b            \n"
        "cnt    v27.16b, v27.16b            \n"
        "add    v20.16b, v20.16b, v24.16b     \n"
        "add    v21.16b, v21.16b, v25.16b     \n"
        "add    v22.16b, v22.16b, v26.16b     \n"
        "add    v23.16b, v23.16b, v27.16b     \n"

        "bne    0b  \n"

        "add    %0, %0, #288    \n"

        "uaddlv h8, v8.16b                  \n"
        "uaddlv h9, v9.16b                  \n"
        "uaddlv h10, v10.16b                  \n"
        "uaddlv h11, v11.16b                  \n"
        "uaddlv h12, v12.16b                  \n"
        "uaddlv h13, v13.16b                  \n"
        "uaddlv h14, v14.16b                  \n"
        "uaddlv h15, v15.16b                  \n"
        "uaddlv h16, v16.16b                  \n"
        "uaddlv h17, v17.16b                  \n"
        "uaddlv h18, v18.16b                  \n"
        "uaddlv h19, v19.16b                  \n"
        "uaddlv h20, v20.16b                  \n"
        "uaddlv h21, v21.16b                  \n"
        "uaddlv h22, v22.16b                  \n"
        "uaddlv h23, v23.16b                  \n"
        "ucvtf  s8, s8                    \n"
        "ucvtf  s9, s9                    \n"
        "ucvtf  s10, s10                    \n"
        "ucvtf  s11, s11                    \n"
        "ucvtf  s12, s12                    \n"
        "ucvtf  s13, s13                    \n"
        "ucvtf  s14, s14                    \n"
        "ucvtf  s15, s15                    \n"
        "ucvtf  s16, s16                    \n"
        "ucvtf  s17, s17                    \n"
        "ucvtf  s18, s18                    \n"
        "ucvtf  s19, s19                    \n"
        "ucvtf  s20, s20                    \n"
        "ucvtf  s21, s21                    \n"
        "ucvtf  s22, s22                    \n"
        "ucvtf  s23, s23                    \n"
        // "str    s8, [%2], #4               \n"
        // "str    s9, [%2], #4               \n"
        // "str    s10, [%2], #4               \n"
        // "str    s11, [%2], #4               \n"
        // "str    s12, [%2], #4               \n"
        // "str    s13, [%2], #4               \n"
        // "str    s14, [%2], #4               \n"
        // "str    s15, [%2], #4               \n"
        // "str    s16, [%2], #4               \n"
        "subs   x0, x0, #1  \n"
        // "str    s17, [%2], #4               \n"
        // "str    s18, [%2], #4               \n"
        // "str    s19, [%2], #4               \n"
        // "str    s20, [%2], #4               \n"
        // "str    s21, [%2], #4               \n"
        // "str    s22, [%2], #4               \n"
        // "str    s23, [%2], #4               \n"
        "bne    1b  \n"

        // "add    %3, %3, #288    \n"

        "subs   %4, %4, #1  \n"
        "bne    2b  \n"

        : "+r"(col_ptr),     // %0
          "+r"(weight_ptr),  // %1
          "+r"(output_ptr),  // %2
          "+r"(nn1),         // %3
          "+r"(nn2)          // %4
        :
        : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8",
          "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18",
          "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28",
          "v29", "v30", "x0", "x1", "x2", "x3");
}

inline void bnn::bconv_3x3_64_opt2(const Mat &bottom_blob, const Mat &weight,
                                   Mat &top_blob, const int pad,
                                   const int stride) {
    /**
     * See bconv_3x3_64_opt4
     */
    static uint64_t col_buf[999999];

    const size_t col_h = weight.h * weight.w;
    const size_t col_w = top_blob.h * top_blob.w * top_blob.c;
    const size_t col_len = col_h * col_w;

    Mat col(col_len, col_buf, DataType::Bit);

    im2col(bottom_blob, 3, 3, pad, pad, stride, stride, 1, 1, col);

    uint64_t *col_ptr = col_buf;
    uint64_t *weight_ptr = static_cast<uint64_t *>(weight.data);
    float *output_ptr = static_cast<float *>(top_blob.data);

    size_t nn1 = top_blob.c;
    size_t nn2 = top_blob.h * top_blob.w;
    asm volatile(
        "1:     \n"
        "ld1    {v0.2d, v1.2d, v2.2d, v3.2d}, [%0], #64     \n"
        "ld1    {v4.1d}, [%0], #8   \n"
        "mov    x1, %1      \n"
        "mov    x0, %3      \n"
        "0:     \n"
        "prfm   pldl1keep, [x1, #128]   \n"
        "ld1    {v5.2d, v6.2d, v7.2d, v8.2d}, [x1], #64     \n"
        "ld1    {v9.1d}, [x1], #8   \n"

        "eor    v15.16b, v0.16b, v5.16b     \n"
        "eor    v16.16b, v1.16b, v6.16b     \n"
        "eor    v17.16b, v2.16b, v7.16b     \n"
        "eor    v18.16b, v3.16b, v8.16b     \n"
        "eor    v19.16b, v4.16b, v9.16b     \n"
        "cnt    v15.16b, v15.16b    \n"
        "cnt    v16.16b, v16.16b    \n"
        "cnt    v17.16b, v17.16b    \n"
        "cnt    v18.16b, v18.16b    \n"
        "cnt    v19.16b, v19.16b    \n"
        "prfm   pldl1keep, [x1, #128]   \n"
        "add    v15.16b, v15.16b, v18.16b   \n"
        "add    v16.16b, v16.16b, v19.16b   \n"
        "add    v15.16b, v15.16b, v16.16b   \n"
        "add    v15.16b, v15.16b, v17.16b   \n"
        "subs   x0, x0, #1  \n"
        "uaddlv h15, v15.16b    \n"
        "ucvtf  s15, s15    \n"
        "str    s15, [%2], #4               \n"
        "bne    0b  \n"
        "subs   %4, %4, #1  \n"
        "bne    1b  \n"

        : "+r"(col_ptr),     // %0
          "+r"(weight_ptr),  // %1
          "+r"(output_ptr),  // %2
          "+r"(nn1),         // %3
          "+r"(nn2)          // %4
        :
        : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8",
          "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18",
          "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28",
          "v29", "v30", "x0", "x1");
}

inline void bnn::bconv_3x3_64_opt4(const Mat &bottom_blob, const Mat &weight,
                                   Mat &top_blob, const int pad,
                                   const int stride) {
    /**
     * This method performs 64-input-channel 3x3 binary conv by
     * im2col + BGEMM.
     *
     * The reason that it outperforms other ways when channel==64
     * is the 128-bit vector registers cannot be fully filled in
     * Binary Direct Convolution + NC1HWC2 memory layout if there
     * are only 64 channels.
     *
     * By contrast, BGEMM can leverage 128-bit registers after im2col,
     * and amortize the memory access.
     *
     */
    // TODO: A more elegant way
    static uint64_t col_buf[999999];

    const size_t col_h = weight.h * weight.w;
    const size_t col_w = top_blob.h * top_blob.w * top_blob.c;
    const size_t col_len = col_h * col_w;

    Mat col(col_len, col_buf, DataType::Bit);

    im2col(bottom_blob, 3, 3, pad, pad, stride, stride, 1, 1, col);

    uint64_t *col_ptr = col_buf;
    float *output_ptr0 = static_cast<float *>(top_blob.data);
    float *output_ptr1 = output_ptr0 + top_blob.c;

    BNN_ASSERT(top_blob.h * top_blob.w % 2 == 0, top_blob.h * top_blob.w);
    BNN_ASSERT(top_blob.c % 2 == 0, top_blob.c);
    size_t nn2 = top_blob.h * top_blob.w >> 1;
    FORZ(i, nn2) {
        size_t nn1 = top_blob.c >> 1;
        uint64_t *weight_ptr = static_cast<uint64_t *>(weight.data);
        asm volatile(
            "ld1    {v0.2d, v1.2d, v2.2d, v3.2d}, [%0], #64     \n"
            "ld1    {v4.1d}, [%0], #8   \n"
            "ld1    {v5.2d, v6.2d, v7.2d, v8.2d}, [%0], #64     \n"
            "ld1    {v9.1d}, [%0], #8   \n"
            "0:     \n"
            "prfm   pldl1keep, [%1, #128]   \n"
            "ld1    {v10.2d, v11.2d, v12.2d, v13.2d}, [%1], #64     \n"
            "ld1    {v14.1d}, [%1], #8   \n"

            "eor    v20.16b, v0.16b, v10.16b     \n"
            "eor    v21.16b, v1.16b, v11.16b     \n"
            "eor    v22.16b, v2.16b, v12.16b     \n"
            "eor    v23.16b, v3.16b, v13.16b     \n"
            "eor    v24.16b, v4.16b, v14.16b     \n"
            "eor    v25.16b, v5.16b, v10.16b     \n"
            "eor    v26.16b, v6.16b, v11.16b     \n"
            "eor    v27.16b, v7.16b, v12.16b     \n"
            "eor    v28.16b, v8.16b, v13.16b     \n"
            "eor    v29.16b, v9.16b, v14.16b     \n"
            "prfm   pldl1keep, [%1, #128]   \n"
            "ld1    {v10.2d, v11.2d, v12.2d, v13.2d}, [%1], #64     \n"
            "ld1    {v14.1d}, [%1], #8   \n"
            "cnt    v20.16b, v20.16b    \n"
            "cnt    v21.16b, v21.16b    \n"
            "cnt    v22.16b, v22.16b    \n"
            "cnt    v23.16b, v23.16b    \n"
            "cnt    v24.16b, v24.16b    \n"
            "add    v20.16b, v20.16b, v23.16b   \n"
            "add    v21.16b, v21.16b, v24.16b   \n"
            "cnt    v25.16b, v25.16b    \n"
            "cnt    v27.16b, v27.16b    \n"
            "add    v20.16b, v20.16b, v21.16b   \n"
            "cnt    v26.16b, v26.16b    \n"
            "cnt    v28.16b, v28.16b    \n"
            "add    v20.16b, v20.16b, v22.16b   \n"
            "cnt    v29.16b, v29.16b    \n"
            "uaddlv h20, v20.16b    \n"
            "add    v25.16b, v25.16b, v28.16b   \n"
            "add    v26.16b, v26.16b, v29.16b   \n"
            "add    v25.16b, v25.16b, v26.16b   \n"
            "add    v25.16b, v25.16b, v27.16b   \n"
            "ucvtf  s20, s20    \n"
            "uaddlv h25, v25.16b    \n"
            "str    s20, [%2], #4               \n"
            // -----
            "eor    v20.16b, v0.16b, v10.16b     \n"
            "ucvtf  s25, s25    \n"
            "eor    v21.16b, v1.16b, v11.16b     \n"
            "eor    v22.16b, v2.16b, v12.16b     \n"
            "str    s25, [%3], #4   \n"
            "eor    v23.16b, v3.16b, v13.16b     \n"
            "eor    v24.16b, v4.16b, v14.16b     \n"
            "eor    v25.16b, v5.16b, v10.16b     \n"
            "eor    v26.16b, v6.16b, v11.16b     \n"
            "eor    v27.16b, v7.16b, v12.16b     \n"
            "eor    v28.16b, v8.16b, v13.16b     \n"
            "eor    v29.16b, v9.16b, v14.16b     \n"
            "cnt    v20.16b, v20.16b    \n"
            "cnt    v21.16b, v21.16b    \n"
            "cnt    v22.16b, v22.16b    \n"
            "cnt    v23.16b, v23.16b    \n"
            "cnt    v24.16b, v24.16b    \n"
            "add    v20.16b, v20.16b, v23.16b   \n"
            "add    v21.16b, v21.16b, v24.16b   \n"
            "cnt    v25.16b, v25.16b    \n"
            "cnt    v27.16b, v27.16b    \n"
            "add    v20.16b, v20.16b, v21.16b   \n"
            "cnt    v26.16b, v26.16b    \n"
            "cnt    v28.16b, v28.16b    \n"
            "add    v20.16b, v20.16b, v22.16b   \n"
            "cnt    v29.16b, v29.16b    \n"
            "subs   %4, %4, #1  \n"
            "uaddlv h20, v20.16b    \n"
            "add    v25.16b, v25.16b, v28.16b   \n"
            "add    v26.16b, v26.16b, v29.16b   \n"
            "add    v25.16b, v25.16b, v26.16b   \n"
            "add    v25.16b, v25.16b, v27.16b   \n"
            "ucvtf  s20, s20    \n"
            "uaddlv h25, v25.16b    \n"
            "str    s20, [%2], #4               \n"
            "ucvtf  s25, s25    \n"
            "str    s25, [%3], #4   \n"
            "bne    0b  \n"

            : "+r"(col_ptr),      // %0
              "+r"(weight_ptr),   // %1
              "+r"(output_ptr0),  // %2
              "+r"(output_ptr1),  // %3
              "+r"(nn1)           // %4
            :
            : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
              "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16",
              "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25",
              "v26", "v27", "v28", "v29", "v30", "x0", "x1");
        output_ptr0 += top_blob.c;
        output_ptr1 += top_blob.c;
    }
}

inline void bnn::bconv_3x3_64_opt(const Mat &bottom_blob, const Mat &weight,
                                  Mat &top_blob) {
    /**
     * See bconv_3x3_64_opt4
     */
    BNN_ASSERT(weight.n % 2 == 0, weight.n);
    FORZ(th, top_blob.h) {
        FORZ(tw, top_blob.w) {
            const auto *bottom_value_0 =
                bottom_blob.point<uint64_t>(th + 0, tw + 0);
            const auto *bottom_value_3 =
                bottom_blob.point<uint64_t>(th + 1, tw + 0);
            const auto *bottom_value_6 =
                bottom_blob.point<uint64_t>(th + 2, tw + 0);
            const auto *w_value_0 = weight.point<uint64_t>(0, 0, 0);
            size_t nn = weight.n / 2;
            auto *top_0 = top_blob.point<uint32_t>(th, tw);
            asm volatile(
                "ld1    {v9.1d}, [%1], #8       \n"
                "ld1    {v10.2d}, [%1] \n"
                "ld1    {v11.1d}, [%2], #8       \n"
                "ld1    {v12.2d}, [%2], #16 \n"
                "ld1    {v13.1d}, [%3], #8       \n"
                "ld1    {v14.2d}, [%3] \n"
                "0:     \n"
                "ld1    {v0.1d}, [%4], #8     \n"
                "ld1    {v1.2d}, [%4], #16     \n"
                "ld1    {v2.1d}, [%4], #8     \n"
                "ld1    {v3.2d}, [%4], #16     \n"
                "ld1    {v4.1d}, [%4], #8     \n"
                "ld1    {v5.2d}, [%4], #16     \n"
                "prfm   pldl1keep, [%4, #128]   \n"

                "eor    v18.16b, v0.16b, v9.16b     \n"
                "eor    v19.16b, v1.16b, v10.16b     \n"
                "eor    v20.16b, v2.16b, v11.16b     \n"
                "eor    v21.16b, v3.16b, v12.16b     \n"
                "eor    v22.16b, v4.16b, v13.16b     \n"
                "eor    v23.16b, v5.16b, v14.16b     \n"
                "cnt    v18.16b, v18.16b    \n"
                "cnt    v19.16b, v19.16b    \n"
                "ld1    {v0.1d}, [%4], #8     \n"
                "ld1    {v1.2d}, [%4], #16     \n"
                "ld1    {v2.1d}, [%4], #8     \n"
                "ld1    {v3.2d}, [%4], #16     \n"
                "ld1    {v4.1d}, [%4], #8     \n"
                "ld1    {v5.2d}, [%4], #16     \n"
                "prfm   pldl1keep, [%4, #128]   \n"
                "cnt    v20.16b, v20.16b    \n"
                "cnt    v21.16b, v21.16b    \n"
                "cnt    v22.16b, v22.16b    \n"
                "cnt    v23.16b, v23.16b    \n"
                "add    v18.16b, v18.16b, v21.16b   \n"
                "add    v19.16b, v19.16b, v22.16b   \n"
                "add    v20.16b, v20.16b, v23.16b   \n"
                "add    v18.16b, v18.16b, v19.16b   \n"
                "add    v18.16b, v18.16b, v20.16b   \n"
                "subs   %0, %0, #1  \n"
                "uaddlv h18, v18.16b    \n"
                "ucvtf  s18, s18    \n"
                "str    s18, [%5], #4               \n"

                "eor    v18.16b, v0.16b, v9.16b     \n"
                "eor    v19.16b, v1.16b, v10.16b     \n"
                "eor    v20.16b, v2.16b, v11.16b     \n"
                "eor    v21.16b, v3.16b, v12.16b     \n"
                "eor    v22.16b, v4.16b, v13.16b     \n"
                "eor    v23.16b, v5.16b, v14.16b     \n"
                "cnt    v18.16b, v18.16b    \n"
                "cnt    v19.16b, v19.16b    \n"
                "cnt    v20.16b, v20.16b    \n"
                "cnt    v21.16b, v21.16b    \n"
                "cnt    v22.16b, v22.16b    \n"
                "cnt    v23.16b, v23.16b    \n"
                "add    v18.16b, v18.16b, v21.16b   \n"
                "add    v19.16b, v19.16b, v22.16b   \n"
                "add    v20.16b, v20.16b, v23.16b   \n"
                "add    v18.16b, v18.16b, v19.16b   \n"
                "add    v18.16b, v18.16b, v20.16b   \n"
                "uaddlv h18, v18.16b    \n"
                "ucvtf  s18, s18    \n"
                "str    s18, [%5], #4               \n"
                "bne    0b  \n"

                : "+r"(nn),              // %0
                  "+r"(bottom_value_0),  // %1
                  "+r"(bottom_value_3),  // %2
                  "+r"(bottom_value_6),  // %3
                  "+r"(w_value_0),       // %4
                  "+r"(top_0)            // %5
                // "+r"(top_1)             // %6
                :
                : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6",
                  "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
                  "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24",
                  "v25", "v26", "v27", "v28", "v29", "v30"

            );
        }
    }
}

inline void bnn::bconv_3x3_64_fallback(const Mat &bottom_blob,
                                       const Mat &weight, Mat &top_blob,
                                       const int stride) {
    FORZ(th, top_blob.h) {
        FORZ(tw, top_blob.w) {
            const auto *bottom_value_0 =
                bottom_blob.point<uint64_t>(th * stride + 0, tw * stride + 0);
            const auto *bottom_value_3 =
                bottom_blob.point<uint64_t>(th * stride + 1, tw * stride + 0);
            const auto *bottom_value_6 =
                bottom_blob.point<uint64_t>(th * stride + 2, tw * stride + 0);
            const auto *w_value_0 = weight.point<uint64_t>(0, 0, 0);
            size_t nn = weight.n;
            auto *top_0 = top_blob.point<uint32_t>(th, tw);
            asm volatile(
                "ld1    {v9.2d}, [%1], #16       \n"
                "ld1    {v10.1d}, [%1] \n"
                "ld1    {v11.2d}, [%2], #16       \n"
                "ld1    {v12.1d}, [%2] \n"
                "ld1    {v13.2d}, [%3], #16       \n"
                "ld1    {v14.1d}, [%3] \n"
                "0:     \n"
                "ld1    {v0.2d}, [%4], #16     \n"
                "ld1    {v1.1d}, [%4], #8     \n"
                "ld1    {v2.2d}, [%4], #16     \n"
                "ld1    {v3.1d}, [%4], #8     \n"
                "ld1    {v4.2d}, [%4], #16     \n"
                "ld1    {v5.1d}, [%4], #8     \n"
                "prfm   pldl1keep, [%4, #128]   \n"

                "eor    v15.16b, v0.16b, v9.16b     \n"
                "eor    v16.16b, v1.16b, v10.16b     \n"
                "eor    v17.16b, v2.16b, v11.16b     \n"
                "eor    v18.16b, v3.16b, v12.16b     \n"
                "eor    v19.16b, v4.16b, v13.16b     \n"
                "eor    v20.16b, v5.16b, v14.16b     \n"
                "cnt    v15.16b, v15.16b    \n"
                "cnt    v16.16b, v16.16b    \n"
                "cnt    v17.16b, v17.16b    \n"
                "cnt    v18.16b, v18.16b    \n"
                "cnt    v19.16b, v19.16b    \n"
                "cnt    v20.16b, v20.16b    \n"
                "add    v15.16b, v15.16b, v18.16b   \n"
                "add    v16.16b, v16.16b, v19.16b   \n"
                "add    v17.16b, v17.16b, v20.16b   \n"
                "add    v15.16b, v15.16b, v16.16b   \n"
                "add    v15.16b, v15.16b, v17.16b   \n"
                "subs   %0, %0, #1  \n"
                "uaddlv h15, v15.16b    \n"
                "ucvtf  s15, s15    \n"
                "str    s15, [%5], #4               \n"
                "bne    0b  \n"

                : "+r"(nn),              // %0
                  "+r"(bottom_value_0),  // %1
                  "+r"(bottom_value_3),  // %2
                  "+r"(bottom_value_6),  // %3
                  "+r"(w_value_0),       // %4
                  "+r"(top_0)            // %5
                :
                : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6",
                  "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
                  "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24",
                  "v25", "v26", "v27", "v28", "v29", "v30"

            );
        }
    }
}

/*
 * weight 3x3, channel 128, output channel 128
 */
inline void bnn::bconv_3x3_128_internal_fallback(
    const uint64_t *bottom_ptr, const int b_w, const uint64_t *weight_ptr,
    float *top_ptr, const int top_h, const int top_w, const int stride) {
#define B(row, col) bottom_ptr + (row)*b_w * 2 + (col)*2
#define W(row, col) weight_ptr + (row)*3 * 2 + (col)*2
#define T(row, col) top_ptr + (row)*top_w * 128 + (col)*128

    FORZ(th, top_h) {
        FORZ(tw, top_w) {
            const auto *bottom_value_0 = B(th * stride + 0, tw * stride + 0);
            const auto *bottom_value_3 = B(th * stride + 1, tw * stride + 0);
            const auto *bottom_value_6 = B(th * stride + 2, tw * stride + 0);
            const auto *w_value_0 = W(0, 0);
            auto *top_0 = T(th, tw);
            size_t nn = 128;
            asm volatile(
                "ld1    {v9.2d, v10.2d, v11.2d}, [%1]               \n"
                "ld1    {v13.2d, v14.2d, v15.2d}, [%2]               \n"
                "ld1    {v17.2d, v18.2d, v19.2d}, [%3]               \n"
                "0:     \n"
                // "prfm   pldl1keep, [%0, #128]     \n"
                "ld1    {v0.2d, v1.2d, v2.2d, v3.2d}, [%0], #64     \n"
                "ld1    {v4.2d, v5.2d, v6.2d, v7.2d}, [%0], #64     \n"
                "ld1    {v8.2d}, [%0], #16                          \n"
                "prfm   pldl1keep, [%0, #128]     \n"
                "prfm   pldl1keep, [%0, #256]     \n"

                "eor    v22.16b, v0.16b, v9.16b      \n"
                "eor    v23.16b, v1.16b, v10.16b      \n"
                "eor    v24.16b, v2.16b, v11.16b      \n"
                "eor    v25.16b, v3.16b, v13.16b      \n"
                "eor    v26.16b, v4.16b, v14.16b      \n"
                "prfm   pldl1keep, [%4]     \n"
                "cnt    v22.16b, v22.16b               \n"
                "cnt    v23.16b, v23.16b               \n"
                "cnt    v24.16b, v24.16b               \n"
                "cnt    v25.16b, v25.16b               \n"
                "cnt    v26.16b, v26.16b               \n"
                "add    v22.16b, v22.16b, v26.16b      \n"
                "add    v23.16b, v23.16b, v25.16b      \n"
                "eor    v26.16b, v5.16b, v15.16b      \n"
                "eor    v27.16b, v6.16b, v17.16b      \n"
                "eor    v28.16b, v7.16b, v18.16b      \n"
                "eor    v29.16b, v8.16b, v19.16b      \n"
                "ldr    s25, [%4]                   \n"
                "cnt    v26.16b, v26.16b               \n"
                "cnt    v27.16b, v27.16b               \n"
                "cnt    v28.16b, v28.16b               \n"
                "cnt    v29.16b, v29.16b               \n"
                "add    v26.16b, v26.16b, v27.16b      \n"
                "add    v28.16b, v28.16b, v29.16b      \n"
                "add    v22.16b, v22.16b, v26.16b      \n"
                "add    v23.16b, v23.16b, v28.16b       \n"
                "add    v22.16b, v22.16b, v24.16b       \n"
                "add    v22.16b, v22.16b, v23.16b       \n"

                "uaddlv   h22, v22.16b                   \n"
                "subs   %5, %5, #1                  \n"
                "ucvtf  s22, s22    \n"
                "fadd   s25, s25, s22   \n"
                "str    s25, [%4], #4               \n"

                "bne    0b                          \n"
                : "+r"(w_value_0),       // %0
                  "+r"(bottom_value_0),  // %1
                  "+r"(bottom_value_3),  // %2
                  "+r"(bottom_value_6),  // %3
                  "+r"(top_0),           // %4
                  "+r"(nn)               // %5
                :
                : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6",
                  "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
                  "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24",
                  "v25", "v26", "v27", "v28", "v29", "v30");
        }
    }
#undef B
#undef W
#undef T
}

/*
 * weight 3x3, stride 1, input channel 128, output channel 128
 */
inline void bnn::bconv_3x3_128_internal_s1(const uint64_t *bottom_ptr,
                                           const int b_w,
                                           const uint64_t *weight_ptr,
                                           float *top_ptr, const int top_h,
                                           const int top_w) {
    BNN_ASSERT(top_w % 2 == 0, top_w);

    const auto *bottom_value_0 = bottom_ptr;
    const auto *bottom_value_3 = bottom_ptr + b_w * 2;
    const auto *bottom_value_6 = bottom_ptr + b_w * 4;
    auto *top_0 = top_ptr;
    auto *top_1 = top_ptr + 128;
    FORZ(th, top_h) {
        FORZS(tw, top_w, 2) {
            const auto *w_value_0 = weight_ptr;
            size_t nn = 128;
            asm volatile(
                "ld1    {v9.2d, v10.2d, v11.2d, v12.2d}, [%1]               \n"
                "ld1    {v13.2d, v14.2d, v15.2d, v16.2d}, [%2]               \n"
                "ld1    {v17.2d, v18.2d, v19.2d, v20.2d}, [%3]               \n"
                "0:     \n"
                // "prfm   pldl1keep, [%0, #128]     \n"
                "ld1    {v0.2d, v1.2d, v2.2d, v3.2d}, [%0], #64     \n"
                "ld1    {v4.2d, v5.2d, v6.2d, v7.2d}, [%0], #64     \n"
                "ld1    {v8.2d}, [%0], #16                          \n"
                "prfm   pldl1keep, [%0, #128]     \n"
                "prfm   pldl1keep, [%0, #256]     \n"

                "eor    v22.16b, v0.16b, v9.16b      \n"
                "eor    v23.16b, v1.16b, v10.16b      \n"
                "eor    v24.16b, v2.16b, v11.16b      \n"
                "eor    v25.16b, v3.16b, v13.16b      \n"
                "eor    v26.16b, v4.16b, v14.16b      \n"
                "prfm   pldl1keep, [%4]     \n"
                "cnt    v22.16b, v22.16b               \n"
                "cnt    v23.16b, v23.16b               \n"
                "cnt    v24.16b, v24.16b               \n"
                "cnt    v25.16b, v25.16b               \n"
                "cnt    v26.16b, v26.16b               \n"
                "add    v22.16b, v22.16b, v26.16b      \n"
                "add    v23.16b, v23.16b, v25.16b      \n"
                "eor    v26.16b, v5.16b, v15.16b      \n"
                "eor    v27.16b, v6.16b, v17.16b      \n"
                "eor    v28.16b, v7.16b, v18.16b      \n"
                "eor    v29.16b, v8.16b, v19.16b      \n"
                "ldr    s25, [%4]                   \n"
                "cnt    v26.16b, v26.16b               \n"
                "cnt    v27.16b, v27.16b               \n"
                "cnt    v28.16b, v28.16b               \n"
                "cnt    v29.16b, v29.16b               \n"
                "add    v26.16b, v26.16b, v27.16b      \n"
                "add    v28.16b, v28.16b, v29.16b      \n"
                "add    v22.16b, v22.16b, v26.16b      \n"
                "add    v23.16b, v23.16b, v28.16b       \n"
                "add    v22.16b, v22.16b, v24.16b       \n"
                "add    v22.16b, v22.16b, v23.16b       \n"

                "uaddlv   h22, v22.16b                   \n"
                "subs   %5, %5, #1                  \n"
                "ucvtf  s22, s22    \n"
                "fadd   s25, s25, s22   \n"
                // *****
                "eor    v22.16b, v0.16b, v10.16b      \n"
                "eor    v23.16b, v1.16b, v11.16b      \n"
                "eor    v24.16b, v2.16b, v12.16b      \n"
                "str    s25, [%4], #4               \n"
                "eor    v25.16b, v3.16b, v14.16b      \n"
                "eor    v26.16b, v4.16b, v15.16b      \n"
                "cnt    v22.16b, v22.16b               \n"
                "cnt    v23.16b, v23.16b               \n"
                "cnt    v24.16b, v24.16b               \n"
                "cnt    v25.16b, v25.16b               \n"
                "cnt    v26.16b, v26.16b               \n"
                "add    v22.16b, v22.16b, v26.16b      \n"
                "eor    v26.16b, v5.16b, v16.16b      \n"
                "eor    v27.16b, v6.16b, v18.16b      \n"
                "eor    v28.16b, v7.16b, v19.16b      \n"
                "eor    v29.16b, v8.16b, v20.16b      \n"
                "cnt    v26.16b, v26.16b               \n"
                "cnt    v27.16b, v27.16b               \n"
                "cnt    v28.16b, v28.16b               \n"
                "cnt    v29.16b, v29.16b               \n"
                "add    v23.16b, v23.16b, v26.16b      \n"
                "ldr    s26, [%6]                   \n"
                "add    v24.16b, v24.16b, v27.16b      \n"
                "add    v25.16b, v25.16b, v28.16b      \n"
                "add    v22.16b, v22.16b, v23.16b   \n"
                "add    v24.16b, v24.16b, v25.16b   \n"
                "add    v22.16b, v22.16b, v24.16b   \n"
                "add    v22.16b, v22.16b, v29.16b   \n"
                "uaddlv h22, v22.16b    \n"
                "ucvtf  s22, s22    \n"
                "fadd   s26, s26, s22   \n"
                "str    s26, [%6], #4               \n"

                "bne    0b                          \n"
                : "+r"(w_value_0),       // %0
                  "+r"(bottom_value_0),  // %1
                  "+r"(bottom_value_3),  // %2
                  "+r"(bottom_value_6),  // %3
                  "+r"(top_0),           // %4
                  "+r"(nn),              // %5
                  "+r"(top_1)            // %6
                :
                : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6",
                  "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
                  "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24",
                  "v25", "v26", "v27", "v28", "v29", "v30");
            bottom_value_0 += 4;
            bottom_value_3 += 4;
            bottom_value_6 += 4;
            top_0 += 128;
            top_1 += 128;
        }
        bottom_value_0 += 4;
        bottom_value_3 += 4;
        bottom_value_6 += 4;
    }
}

inline void pack_weight_3x3(int num_output, int num_input, uint64_t *a,
                            uint64_t *b) {
#define A(i, k, j) \
    a[(i)*9 * num_input + k * num_input + (j)]  // A(output, s, input)
    BNN_ASSERT(num_input, 4);
    FORZS(k, num_input, 2) {
        FORZ(j, num_output) {
            FORZ(i, 9) {
                *b++ = A(j, i, k + 0);
                *b++ = A(j, i, k + 1);
            }
        }
    }
#undef A
}

inline void pack_input_3x3(uint64_t *a, int width, int height, int channels,
                           uint64_t *b) {
#define A(i, j, k) \
    a[(i)*width * channels + (j)*channels + (k)]  // A(row, col, channel)
    BNN_ASSERT(channels, 4);

    FORZS(i, channels, 2) {
        FORZ(j, height) {
            FORZ(k, width) {
                *b++ = A(j, k, i + 0);
                *b++ = A(j, k, i + 1);
            }
        }
    }

#undef A
}

inline void unpack_output(float *b, float *a, int width, int height,
                          int channels) {
#define A(i, j, k) \
    a[(i)*width * channels + (j)*channels + (k)]  // A(row, col, channel)

    FORZS(k, channels, 128) {
        FORZ(h, height) {
            FORZ(w, width) {
                FORZ(i, 128) { A(h, w, k + i) = *b++; }
            }
        }
    }

#undef A
}
#endif // __aarch64__

inline void bnn::bconv_3x3(const Mat &bottom_blob, const Mat &weight,
                           Mat &top_blob, const int stride) {
    /**
     * This method shows our NC1HWC2 memory layout and Binary
     * Direct Convolution. The input tensor and weight is packed
     * into NC1HWC2 layout (in the method `pack_weight_3x3` and
     * `pack_input_3x3`), the spatial redundancy is then leveraged
     * in `bconv_3x3_128_internal_s1`.
     */
#ifdef __aarch64__
    // TODO: more elegant way
    static uint64_t packed_weight[999999];
    static uint64_t packed_input[9999999];
    static float packed_output[9999999];
    BNN_ASSERT(bottom_blob.total() < 9999999, bottom_blob.total());
    BNN_ASSERT(weight.total() < 999999, weight.total());

    if (bottom_blob.c == 1) {
        bconv_3x3_64(bottom_blob, weight, top_blob, stride);
    } else if (bottom_blob.c == 2 && top_blob.c == 128) {
        top_blob.fill<float>(0.f);
        if (stride == 1 && top_blob.w % 2 == 0) {
            bconv_3x3_128_internal_s1(
                static_cast<uint64_t *>(bottom_blob.data), bottom_blob.w,
                static_cast<uint64_t *>(weight.data),
                static_cast<float *>(top_blob.data), top_blob.h, top_blob.w);
        } else {
            bconv_3x3_128_internal_fallback(
                static_cast<uint64_t *>(bottom_blob.data), bottom_blob.w,
                static_cast<uint64_t *>(weight.data),
                static_cast<float *>(top_blob.data), top_blob.h, top_blob.w,
                stride);
        }
    } else {
        BNN_ASSERT(top_blob.c % 128 == 0, top_blob.c);
        pack_weight_3x3(weight.n, weight.c,
                        static_cast<uint64_t *>(weight.data), packed_weight);
        pack_input_3x3(static_cast<uint64_t *>(bottom_blob.data), bottom_blob.w,
                       bottom_blob.h, bottom_blob.c, packed_input);

        const int th = top_blob.h;
        const int tw = top_blob.w;

        const int bw = bottom_blob.w;

        const int bc_128_num = bottom_blob.c / 2;
        const int tc_128_num = top_blob.c / 128;

        const int input_blocks = bottom_blob.total() / bc_128_num;
        const int weight_blocks = weight.total() / bc_128_num / tc_128_num;
        const int output_blocks = top_blob.total() / tc_128_num;

        memset(packed_output, 0, top_blob.total() * top_blob.elemsize);

        FORZ(i, bc_128_num) {
            FORZ(j, tc_128_num) {
                if (stride == 1 && tw % 2 == 0) {
                    bconv_3x3_128_internal_s1(
                        packed_input + input_blocks * i, bw,
                        packed_weight + weight_blocks * (i * tc_128_num + j),
                        packed_output + output_blocks * j, th, tw);
                } else {
                    bconv_3x3_128_internal_fallback(
                        packed_input + input_blocks * i, bw,
                        packed_weight + weight_blocks * (i * tc_128_num + j),
                        packed_output + output_blocks * j, th, tw, stride);
                }
            }
        }
        unpack_output(packed_output, static_cast<float *>(top_blob.data),
                      top_blob.w, top_blob.h, top_blob.c);
    }
#else // __aarch64__
    baseline_bconv(bottom_blob, weight, 3, 3, 0, 0, stride, stride, 1, 1, top_blob.c, top_blob);
#endif // __aarch64__
}

#ifdef __aarch64__
inline void bnn::bconv_1x1_512(const Mat &bottom_blob, const Mat &weight,
                               Mat &top_blob) {
    FORZS(th, top_blob.h, 2) {
        FORZ(tw, top_blob.w) {
            const auto *bottom_value_0 =
                bottom_blob.point<uint64_t>(th + 0, tw);
            const auto *bottom_value_1 =
                bottom_blob.point<uint64_t>(th + 1, tw);

            const auto *w_value_0 = weight.point<uint64_t>(0, 0, 0);
            auto *top_0 = top_blob.point<uint32_t>(th + 0, tw);
            auto *top_1 = top_blob.point<uint32_t>(th + 1, tw);
            size_t nn = weight.n >> 2;
            asm volatile(
                "prfm   pldl1keep, [%3, #128]     \n"
                "prfm   pldl1keep, [%4, #128]     \n"
                "ld1	{v23.2d, v24.2d, v25.2d, v26.2d}, [%2]       \n"
                "ld1	{v27.2d, v28.2d, v29.2d, v30.2d}, [%3]       \n"
                "0: \n"

                // ----------------------
                // ----------------------

                "ld1    {v4.2d, v5.2d, v6.2d, v7.2d}, [%4], #64      \n"

                "eor	v0.16b, v23.16b, v4.16b    \n"
                "eor	v1.16b, v24.16b, v5.16b    \n"
                "eor	v2.16b, v25.16b, v6.16b    \n"
                "eor	v3.16b, v26.16b, v7.16b    \n"

                "cnt	v0.16b, v0.16b        \n"
                "cnt	v1.16b, v1.16b        \n"
                "cnt	v2.16b, v2.16b        \n"
                "cnt	v3.16b, v3.16b        \n"

                "prfm   pldl1keep, [%4, #128]     \n"
                "ld1    {v8.2d, v9.2d, v10.2d, v11.2d}, [%4], #64      \n"

                "add    v0.16b, v1.16b, v0.16b            \n"
                "add    v2.16b, v3.16b, v2.16b            \n"

                "addv   b0, v0.16b                          \n"
                "addv   b2, v2.16b                          \n"
                "add    d0, d0, d2                  \n"
                "ins    v20.s[0], v0.s[0]                 \n"

                // --------------

                "eor	v0.16b, v23.16b, v8.16b    \n"
                "eor	v1.16b, v24.16b, v9.16b    \n"
                "eor	v2.16b, v25.16b, v10.16b    \n"
                "eor	v3.16b, v26.16b, v11.16b    \n"

                "cnt	v0.16b, v0.16b        \n"
                "cnt	v1.16b, v1.16b        \n"
                "cnt	v2.16b, v2.16b        \n"
                "cnt	v3.16b, v3.16b        \n"

                "prfm   pldl1keep, [%4, #128]     \n"
                "ld1    {v12.2d, v13.2d, v14.2d, v15.2d}, [%4], #64      \n"

                "add    v0.16b, v1.16b, v0.16b            \n"
                "add    v2.16b, v3.16b, v2.16b            \n"

                "addv   b0, v0.16b                          \n"
                "addv   b2, v2.16b                          \n"
                "add    d0, d0, d2                  \n"
                "ins    v20.s[1], v0.s[0]                 \n"

                // --------------

                "eor	v0.16b, v23.16b, v12.16b    \n"
                "eor	v1.16b, v24.16b, v13.16b    \n"
                "eor	v2.16b, v25.16b, v14.16b    \n"
                "eor	v3.16b, v26.16b, v15.16b    \n"

                "cnt	v0.16b, v0.16b        \n"
                "cnt	v1.16b, v1.16b        \n"
                "cnt	v2.16b, v2.16b        \n"
                "cnt	v3.16b, v3.16b        \n"

                "prfm   pldl1keep, [%4, #128]     \n"
                "ld1    {v16.2d, v17.2d, v18.2d, v19.2d}, [%4], #64      \n"

                "add    v0.16b, v1.16b, v0.16b            \n"
                "add    v2.16b, v3.16b, v2.16b            \n"

                "addv   b0, v0.16b                          \n"
                "addv   b2, v2.16b                          \n"
                "add    d0, d0, d2                  \n"
                "ins    v20.s[2], v0.s[0]                 \n"

                // --------------

                "subs   %5, %5, #1           \n"

                "eor	v0.16b, v23.16b, v16.16b    \n"
                "eor	v1.16b, v24.16b, v17.16b    \n"
                "eor	v2.16b, v25.16b, v18.16b    \n"
                "eor	v3.16b, v26.16b, v19.16b    \n"

                "cnt	v0.16b, v0.16b        \n"
                "cnt	v1.16b, v1.16b        \n"
                "cnt	v2.16b, v2.16b        \n"
                "cnt	v3.16b, v3.16b        \n"

                "add    v0.16b, v1.16b, v0.16b            \n"
                "add    v2.16b, v3.16b, v2.16b            \n"

                "addv   b0, v0.16b                          \n"
                "addv   b2, v2.16b                          \n"
                "add    d0, d0, d2                  \n"
                "ins    v20.s[3], v0.s[0]                 \n"

                "prfm   pldl1keep, [%0, #128]     \n"
                "st1    {v20.4s}, [%0], #16        \n"

                // ----------------------
                // ----------------------

                "eor	v0.16b, v27.16b, v4.16b    \n"
                "eor	v1.16b, v28.16b, v5.16b    \n"
                "eor	v2.16b, v29.16b, v6.16b    \n"
                "eor	v3.16b, v30.16b, v7.16b    \n"

                "cnt	v0.16b, v0.16b        \n"
                "cnt	v1.16b, v1.16b        \n"
                "cnt	v2.16b, v2.16b        \n"
                "cnt	v3.16b, v3.16b        \n"

                "add    v0.16b, v1.16b, v0.16b            \n"
                "add    v2.16b, v3.16b, v2.16b            \n"

                "addv   b0, v0.16b                          \n"
                "addv   b2, v2.16b                          \n"
                "add    d0, d0, d2                  \n"
                "ins    v20.s[0], v0.s[0]                 \n"

                // --------------

                "eor	v0.16b, v27.16b, v8.16b    \n"
                "eor	v1.16b, v28.16b, v9.16b    \n"
                "eor	v2.16b, v29.16b, v10.16b    \n"
                "eor	v3.16b, v30.16b, v11.16b    \n"

                "cnt	v0.16b, v0.16b        \n"
                "cnt	v1.16b, v1.16b        \n"
                "cnt	v2.16b, v2.16b        \n"
                "cnt	v3.16b, v3.16b        \n"

                "add    v0.16b, v1.16b, v0.16b            \n"
                "add    v2.16b, v3.16b, v2.16b            \n"

                "addv   b0, v0.16b                          \n"
                "addv   b2, v2.16b                          \n"
                "add    d0, d0, d2                  \n"
                "ins    v20.s[1], v0.s[0]                 \n"

                // --------------

                "eor	v0.16b, v27.16b, v12.16b    \n"
                "eor	v1.16b, v28.16b, v13.16b    \n"
                "eor	v2.16b, v29.16b, v14.16b    \n"
                "eor	v3.16b, v30.16b, v15.16b    \n"

                "cnt	v0.16b, v0.16b        \n"
                "cnt	v1.16b, v1.16b        \n"
                "cnt	v2.16b, v2.16b        \n"
                "cnt	v3.16b, v3.16b        \n"

                "add    v0.16b, v1.16b, v0.16b            \n"
                "add    v2.16b, v3.16b, v2.16b            \n"

                "addv   b0, v0.16b                          \n"
                "addv   b2, v2.16b                          \n"
                "add    d0, d0, d2                  \n"
                "ins    v20.s[2], v0.s[0]                 \n"

                // --------------

                "eor	v0.16b, v27.16b, v16.16b    \n"
                "eor	v1.16b, v28.16b, v17.16b    \n"
                "eor	v2.16b, v29.16b, v18.16b    \n"
                "eor	v3.16b, v30.16b, v19.16b    \n"

                "cnt	v0.16b, v0.16b        \n"
                "cnt	v1.16b, v1.16b        \n"
                "cnt	v2.16b, v2.16b        \n"
                "cnt	v3.16b, v3.16b        \n"

                "add    v0.16b, v1.16b, v0.16b            \n"
                "add    v2.16b, v3.16b, v2.16b            \n"

                "addv   b0, v0.16b                          \n"
                "addv   b2, v2.16b                          \n"
                "add    d0, d0, d2                  \n"
                "ins    v20.s[3], v0.s[0]                 \n"

                "prfm   pldl1keep, [%1, #128]     \n"
                "st1    {v20.4s}, [%1], #16        \n"

                "bne    0b              \n"
                : "+r"(top_0),           // %0
                  "+r"(top_1),           // %1
                  "+r"(bottom_value_0),  // %2
                  "+r"(bottom_value_1),  // %3
                  "+r"(w_value_0),       // %4
                  "+r"(nn),              // %5
                  "=m"(*(uint32_t(*)[4 * weight.n]) top_0),
                  "=m"(*(uint32_t(*)[4 * weight.n]) top_1)
                : "m"(*(uint64_t(*)[8])bottom_value_0),
                  "m"(*(uint64_t(*)[8])bottom_value_1),
                  "m"(*(uint64_t(*)[8 * weight.n]) w_value_0)

                : "cc", "x9", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
                  "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16",
                  "v17", "v18", "v19", "v20", "v23", "v24", "v25", "v26", "v27",
                  "v28", "v29", "v30");
        }
    }
}

inline void bnn::bconv_1x1_256(const Mat &bottom_blob, const Mat &weight,
                               Mat &top_blob) {
    FORZS(th, top_blob.h, 4) {
        FORZ(tw, top_blob.w) {
            const auto *bottom_value_0 =
                bottom_blob.point<uint64_t>(th + 0, tw);
            const auto *bottom_value_1 =
                bottom_blob.point<uint64_t>(th + 1, tw);
            const auto *bottom_value_2 =
                bottom_blob.point<uint64_t>(th + 2, tw);
            const auto *bottom_value_3 =
                bottom_blob.point<uint64_t>(th + 3, tw);

            const auto *w_value_0 = weight.point<uint64_t>(0, 0, 0);
            auto *top_0 = top_blob.point<uint32_t>(th + 0, tw);
            auto *top_1 = top_blob.point<uint32_t>(th + 1, tw);
            auto *top_2 = top_blob.point<uint32_t>(th + 2, tw);
            auto *top_3 = top_blob.point<uint32_t>(th + 3, tw);

            size_t nn = weight.n >> 2;
            asm volatile(
                "ld1	{v23.2d, v24.2d}, [%4]       \n"
                "ld1	{v25.2d, v26.2d}, [%5]       \n"
                "ld1	{v27.2d, v28.2d}, [%6]       \n"
                "ld1	{v29.2d, v30.2d}, [%7]       \n"
                "0: \n"
                "prfm   pldl1keep, [%8, #128]     \n"
                "ld1    {v15.2d, v16.2d, v17.2d, v18.2d}, [%8], #64      \n"

                // ----------

                "eor	v0.16b, v23.16b, v15.16b    \n"
                "eor	v1.16b, v24.16b, v16.16b    \n"
                "eor	v2.16b, v23.16b, v17.16b    \n"
                "eor	v3.16b, v24.16b, v18.16b    \n"

                "cnt	v0.16b, v0.16b        \n"
                "cnt	v1.16b, v1.16b        \n"
                "cnt	v2.16b, v2.16b        \n"
                "cnt	v3.16b, v3.16b        \n"

                "prfm   pldl1keep, [%8, #128]     \n"
                "ld1    {v19.2d, v20.2d, v21.2d, v22.2d}, [%8], #64      \n"

                "eor	v4.16b, v23.16b, v19.16b    \n"
                "eor	v5.16b, v24.16b, v20.16b    \n"
                "eor	v6.16b, v23.16b, v21.16b    \n"
                "eor	v7.16b, v24.16b, v22.16b    \n"

                "cnt	v4.16b, v4.16b        \n"
                "cnt	v5.16b, v5.16b        \n"
                "cnt	v6.16b, v6.16b        \n"
                "cnt	v7.16b, v7.16b        \n"

                "add    v0.16b, v1.16b, v0.16b            \n"
                "add    v2.16b, v3.16b, v2.16b            \n"
                "add    v4.16b, v5.16b, v4.16b            \n"
                "add    v6.16b, v7.16b, v6.16b            \n"

                "addv	b0, v0.16b       \n"
                "addv	b1, v2.16b       \n"
                "addv	b2, v4.16b       \n"
                "addv	b3, v6.16b       \n"

                "subs   %9, %9, #1           \n"
                "ins    v12.s[0], v0.s[0]                 \n"
                "ins    v12.s[1], v1.s[0]                 \n"
                "ins    v12.s[2], v2.s[0]                 \n"
                "ins    v12.s[3], v3.s[0]                 \n"

                // ----------

                "eor	v0.16b, v25.16b, v15.16b    \n"
                "eor	v1.16b, v26.16b, v16.16b    \n"
                "eor	v2.16b, v25.16b, v17.16b    \n"
                "eor	v3.16b, v26.16b, v18.16b    \n"

                "cnt	v0.16b, v0.16b        \n"
                "cnt	v1.16b, v1.16b        \n"
                "cnt	v2.16b, v2.16b        \n"
                "cnt	v3.16b, v3.16b        \n"

                "eor	v4.16b, v25.16b, v19.16b    \n"
                "eor	v5.16b, v26.16b, v20.16b    \n"
                "eor	v6.16b, v25.16b, v21.16b    \n"
                "eor	v7.16b, v26.16b, v22.16b    \n"

                "cnt	v4.16b, v4.16b        \n"
                "cnt	v5.16b, v5.16b        \n"
                "cnt	v6.16b, v6.16b        \n"
                "cnt	v7.16b, v7.16b        \n"

                "add    v0.16b, v1.16b, v0.16b            \n"
                "add    v2.16b, v3.16b, v2.16b            \n"
                "add    v4.16b, v5.16b, v4.16b            \n"
                "add    v6.16b, v7.16b, v6.16b            \n"

                "addv	b0, v0.16b       \n"
                "addv	b1, v2.16b       \n"
                "addv	b2, v4.16b       \n"
                "addv	b3, v6.16b       \n"

                "ins    v13.s[0], v0.s[0]                 \n"
                "ins    v13.s[1], v1.s[0]                 \n"
                "ins    v13.s[2], v2.s[0]                 \n"
                "ins    v13.s[3], v3.s[0]                 \n"

                // ---------------------

                "eor	v0.16b, v27.16b, v15.16b    \n"
                "eor	v1.16b, v28.16b, v16.16b    \n"
                "eor	v2.16b, v27.16b, v17.16b    \n"
                "eor	v3.16b, v28.16b, v18.16b    \n"

                "cnt	v0.16b, v0.16b        \n"
                "cnt	v1.16b, v1.16b        \n"
                "cnt	v2.16b, v2.16b        \n"
                "cnt	v3.16b, v3.16b        \n"

                "eor	v4.16b, v27.16b, v19.16b    \n"
                "eor	v5.16b, v28.16b, v20.16b    \n"
                "eor	v6.16b, v27.16b, v21.16b    \n"
                "eor	v7.16b, v28.16b, v22.16b    \n"

                "cnt	v4.16b, v4.16b        \n"
                "cnt	v5.16b, v5.16b        \n"
                "cnt	v6.16b, v6.16b        \n"
                "cnt	v7.16b, v7.16b        \n"

                "add    v0.16b, v1.16b, v0.16b            \n"
                "add    v2.16b, v3.16b, v2.16b            \n"
                "add    v4.16b, v5.16b, v4.16b            \n"
                "add    v6.16b, v7.16b, v6.16b            \n"

                "addv	b0, v0.16b       \n"
                "addv	b1, v2.16b       \n"
                "addv	b2, v4.16b       \n"
                "addv	b3, v6.16b       \n"

                "ins    v14.s[0], v0.s[0]                 \n"
                "ins    v14.s[1], v1.s[0]                 \n"
                "ins    v14.s[2], v2.s[0]                 \n"
                "ins    v14.s[3], v3.s[0]                 \n"

                // ---------------------

                "eor	v0.16b, v29.16b, v15.16b    \n"
                "eor	v1.16b, v30.16b, v16.16b    \n"
                "eor	v2.16b, v29.16b, v17.16b    \n"
                "eor	v3.16b, v30.16b, v18.16b    \n"

                "cnt	v0.16b, v0.16b        \n"
                "cnt	v1.16b, v1.16b        \n"
                "cnt	v2.16b, v2.16b        \n"
                "cnt	v3.16b, v3.16b        \n"

                "eor	v4.16b, v29.16b, v19.16b    \n"
                "eor	v5.16b, v30.16b, v20.16b    \n"
                "eor	v6.16b, v29.16b, v21.16b    \n"
                "eor	v7.16b, v30.16b, v22.16b    \n"

                "cnt	v4.16b, v4.16b        \n"
                "cnt	v5.16b, v5.16b        \n"
                "cnt	v6.16b, v6.16b        \n"
                "cnt	v7.16b, v7.16b        \n"

                "add    v0.16b, v1.16b, v0.16b            \n"
                "add    v2.16b, v3.16b, v2.16b            \n"
                "add    v4.16b, v5.16b, v4.16b            \n"
                "add    v6.16b, v7.16b, v6.16b            \n"

                "addv	b0, v0.16b       \n"
                "addv	b1, v2.16b       \n"
                "addv	b2, v4.16b       \n"
                "addv	b3, v6.16b       \n"

                "ins    v15.s[0], v0.s[0]                 \n"
                "ins    v15.s[1], v1.s[0]                 \n"
                "ins    v15.s[2], v2.s[0]                 \n"
                "ins    v15.s[3], v3.s[0]                 \n"

                // ---------------------

                "prfm   pldl1keep, [%0, #128]     \n"
                "st1    {v12.4s}, [%0], #16        \n"
                "prfm   pldl1keep, [%1, #128]     \n"
                "st1    {v13.4s}, [%1], #16        \n"
                "prfm   pldl1keep, [%0, #128]     \n"
                "st1    {v14.4s}, [%2], #16        \n"
                "prfm   pldl1keep, [%1, #128]     \n"
                "st1    {v15.4s}, [%3], #16        \n"

                "bne    0b              \n"
                : "+r"(top_0),           // %0
                  "+r"(top_1),           // %1
                  "+r"(top_2),           // %2
                  "+r"(top_3),           // %3
                  "+r"(bottom_value_0),  // %4
                  "+r"(bottom_value_1),  // %5
                  "+r"(bottom_value_2),  // %6
                  "+r"(bottom_value_3),  // %7
                  "+r"(w_value_0),       // %8
                  "+r"(nn)               // %9
                :
                : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6",
                  "v7", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17",
                  "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26",
                  "v27", "v28", "v29", "v30");
        }
    }
}

inline void bnn::bconv_1x1_128(const Mat &bottom_blob, const Mat &weight,
                               Mat &top_blob) {
#define UNROLL_1x1_128 1
#if UNROLL_1x1_128
    FORZS(th, top_blob.h, 4) {
        const auto *bottom_value_0 = bottom_blob.point<uint64_t>(th + 0, 0);
        const auto *bottom_value_1 = bottom_blob.point<uint64_t>(th + 1, 0);
        const auto *bottom_value_2 = bottom_blob.point<uint64_t>(th + 2, 0);
        const auto *bottom_value_3 = bottom_blob.point<uint64_t>(th + 3, 0);

        FORZ(tw, top_blob.w) {
            const auto *w_value_0 = weight.point<uint64_t>(0, 0, 0);
            auto *top_0 = top_blob.point<float>(th + 0, tw);
            auto *top_1 = top_blob.point<float>(th + 1, tw);
            auto *top_2 = top_blob.point<float>(th + 2, tw);
            auto *top_3 = top_blob.point<float>(th + 3, tw);
            size_t nn = weight.n >> 2;
            asm volatile(
                "ld1	{v27.2d}, [%4]       \n"
                "ld1	{v28.2d}, [%5]       \n"
                "ld1	{v29.2d}, [%6]       \n"
                "ld1	{v30.2d}, [%7]       \n"
                "0: \n"
                "prfm   pldl1keep, [%8, #128]     \n"
                "ld1    {v16.2d, v17.2d, v18.2d, v19.2d}, [%8], #64      \n"

#define BNN_TYPE 0  // group by type
#if BNN_TYPE
                "eor	v0.16b, v27.16b, v16.16b    \n"
                "eor	v1.16b, v27.16b, v17.16b    \n"
                "eor	v2.16b, v27.16b, v18.16b    \n"
                "eor	v3.16b, v27.16b, v19.16b    \n"

                "eor	v4.16b, v28.16b, v16.16b    \n"
                "eor	v5.16b, v28.16b, v17.16b    \n"
                "eor	v6.16b, v28.16b, v18.16b    \n"
                "eor	v7.16b, v28.16b, v19.16b    \n"

                "eor	v8.16b, v29.16b, v16.16b    \n"
                "eor	v9.16b, v29.16b, v17.16b    \n"
                "eor	v10.16b, v29.16b, v18.16b    \n"
                "eor	v11.16b, v29.16b, v19.16b    \n"

                "eor	v12.16b, v30.16b, v16.16b    \n"
                "eor	v13.16b, v30.16b, v17.16b    \n"
                "eor	v14.16b, v30.16b, v18.16b    \n"
                "eor	v15.16b, v30.16b, v19.16b    \n"

                "cnt	v0.16b, v0.16b        \n"
                "cnt	v1.16b, v1.16b        \n"
                "cnt	v2.16b, v2.16b        \n"
                "cnt	v3.16b, v3.16b        \n"

                "cnt	v4.16b, v4.16b        \n"
                "cnt	v5.16b, v5.16b        \n"
                "cnt	v6.16b, v6.16b        \n"
                "cnt	v7.16b, v7.16b        \n"

                "cnt	v8.16b, v8.16b        \n"
                "cnt	v9.16b, v9.16b        \n"
                "cnt	v10.16b, v10.16b        \n"
                "cnt	v11.16b, v11.16b        \n"

                "cnt	v12.16b, v12.16b        \n"
                "cnt	v13.16b, v13.16b        \n"
                "cnt	v14.16b, v14.16b        \n"
                "cnt	v15.16b, v15.16b        \n"

                "addv	b0, v0.16b       \n"
                "addv	b1, v1.16b       \n"
                "addv	b2, v2.16b       \n"
                "addv	b3, v3.16b       \n"

                "addv	b4, v4.16b       \n"
                "addv	b5, v5.16b       \n"
                "addv	b6, v6.16b       \n"
                "addv	b7, v7.16b       \n"

                "addv	b8, v8.16b       \n"
                "addv	b9, v9.16b       \n"
                "addv	b10, v10.16b       \n"
                "addv	b11, v11.16b       \n"

                "addv	b12, v12.16b       \n"
                "addv	b13, v13.16b       \n"
                "addv	b14, v14.16b       \n"
                "addv	b15, v15.16b       \n"

                "subs   %9, %9, #1           \n"

                "ins    v20.s[0], v0.s[0]                 \n"
                "ins    v20.s[1], v1.s[0]                 \n"
                "ins    v20.s[2], v2.s[0]                 \n"
                "ins    v20.s[3], v3.s[0]                 \n"
                "ucvtf  v20.4s, v20.4s     \n"

                "ins    v21.s[0], v4.s[0]                 \n"
                "ins    v21.s[1], v5.s[0]                 \n"
                "ins    v21.s[2], v6.s[0]                 \n"
                "ins    v21.s[3], v7.s[0]                 \n"
                "ucvtf  v21.4s, v21.4s     \n"

                "ins    v22.s[0], v8.s[0]                 \n"
                "ins    v22.s[1], v9.s[0]                 \n"
                "ins    v22.s[2], v10.s[0]                 \n"
                "ins    v22.s[3], v11.s[0]                 \n"
                "ucvtf  v22.4s, v22.4s     \n"

                "ins    v23.s[0], v12.s[0]                 \n"
                "ins    v23.s[1], v13.s[0]                 \n"
                "ins    v23.s[2], v14.s[0]                 \n"
                "ins    v23.s[3], v15.s[0]                 \n"
                "ucvtf  v23.4s, v23.4s     \n"

                "prfm   pldl1keep, [%0, #128]     \n"
                "st1    {v20.4s}, [%0], #16        \n"
                "prfm   pldl1keep, [%1, #128]     \n"
                "st1    {v21.4s}, [%1], #16        \n"
                "prfm   pldl1keep, [%2, #128]     \n"
                "st1    {v22.4s}, [%2], #16        \n"
                "prfm   pldl1keep, [%3, #128]     \n"
                "st1    {v23.4s}, [%3], #16        \n"

#else   // BNN_TYPE

                "eor	v0.16b, v27.16b, v16.16b    \n"
                "eor	v1.16b, v27.16b, v17.16b    \n"
                "eor	v2.16b, v27.16b, v18.16b    \n"
                "eor	v3.16b, v27.16b, v19.16b    \n"

                "cnt	v0.16b, v0.16b        \n"
                "cnt	v1.16b, v1.16b        \n"
                "cnt	v2.16b, v2.16b        \n"
                "cnt	v3.16b, v3.16b        \n"

                "addv	b0, v0.16b       \n"
                "addv	b1, v1.16b       \n"
                "addv	b2, v2.16b       \n"
                "addv	b3, v3.16b       \n"

                "ins    v20.s[0], v0.s[0]                 \n"
                "ins    v20.s[1], v1.s[0]                 \n"
                "ins    v20.s[2], v2.s[0]                 \n"
                "ins    v20.s[3], v3.s[0]                 \n"

                "prfm   pldl1keep, [%0, #128]     \n"  // Sparse the storing,
                                                       // avoid stall

                "eor	v0.16b, v28.16b, v16.16b    \n"
                "eor	v1.16b, v28.16b, v17.16b    \n"
                "eor	v2.16b, v28.16b, v18.16b    \n"
                "eor	v3.16b, v28.16b, v19.16b    \n"

                "ucvtf v20.4s, v20.4s  \n"

                "cnt	v0.16b, v0.16b        \n"
                "cnt	v1.16b, v1.16b        \n"
                "cnt	v2.16b, v2.16b        \n"
                "cnt	v3.16b, v3.16b        \n"

                "st1    {v20.4s}, [%0], #16        \n"

                "addv	b0, v0.16b       \n"
                "addv	b1, v1.16b       \n"
                "addv	b2, v2.16b       \n"
                "addv	b3, v3.16b       \n"

                "prfm   pldl1keep, [%1, #128]     \n"

                "ins    v20.s[0], v0.s[0]                 \n"
                "ins    v20.s[1], v1.s[0]                 \n"
                "ins    v20.s[2], v2.s[0]                 \n"
                "ins    v20.s[3], v3.s[0]                 \n"

                "eor	v0.16b, v29.16b, v16.16b    \n"
                "eor	v1.16b, v29.16b, v17.16b    \n"
                "eor	v2.16b, v29.16b, v18.16b    \n"
                "eor	v3.16b, v29.16b, v19.16b    \n"

                "ucvtf v20.4s, v20.4s  \n"

                "cnt	v0.16b, v0.16b        \n"
                "cnt	v1.16b, v1.16b        \n"
                "cnt	v2.16b, v2.16b        \n"
                "cnt	v3.16b, v3.16b        \n"

                "st1    {v20.4s}, [%1], #16        \n"

                "addv	b0, v0.16b       \n"
                "addv	b1, v1.16b       \n"
                "addv	b2, v2.16b       \n"
                "addv	b3, v3.16b       \n"

                "prfm   pldl1keep, [%2, #128]     \n"

                "ins    v20.s[0], v0.s[0]                 \n"
                "ins    v20.s[1], v1.s[0]                 \n"
                "ins    v20.s[2], v2.s[0]                 \n"
                "ins    v20.s[3], v3.s[0]                 \n"

                "eor	v0.16b, v30.16b, v16.16b    \n"
                "eor	v1.16b, v30.16b, v17.16b    \n"
                "eor	v2.16b, v30.16b, v18.16b    \n"
                "eor	v3.16b, v30.16b, v19.16b    \n"

                "prfm   pldl1keep, [%3, #128]     \n"

                "ucvtf v20.4s, v20.4s  \n"

                "cnt	v0.16b, v0.16b        \n"
                "cnt	v1.16b, v1.16b        \n"
                "cnt	v2.16b, v2.16b        \n"
                "cnt	v3.16b, v3.16b        \n"

                "st1    {v20.4s}, [%2], #16        \n"

                "addv	b0, v0.16b       \n"
                "addv	b1, v1.16b       \n"
                "addv	b2, v2.16b       \n"
                "addv	b3, v3.16b       \n"

                "subs   %9, %9, #1           \n"

                "ins    v20.s[0], v0.s[0]                 \n"
                "ins    v20.s[1], v1.s[0]                 \n"
                "ins    v20.s[2], v2.s[0]                 \n"
                "ins    v20.s[3], v3.s[0]                 \n"
                "ucvtf v20.4s, v20.4s  \n"

                "st1    {v20.4s}, [%3], #16        \n"
#endif  // BNN_TYPE

                "bne    0b              \n"
                : "+r"(top_0),           // %0
                  "+r"(top_1),           // %1
                  "+r"(top_2),           // %2
                  "+r"(top_3),           // %3
                  "+r"(bottom_value_0),  // %4
                  "+r"(bottom_value_1),  // %5
                  "+r"(bottom_value_2),  // %6
                  "+r"(bottom_value_3),  // %7
                  "+r"(w_value_0),       // %8
                  "+r"(nn)               // %9
                :
#if BNN_TYPE
                : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6",
                  "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
                  "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v27",
                  "v28", "v29", "v30"
#else   // BNN_TYPE
                : "cc", "memory", "v0", "v1", "v2", "v3", "v15", "v16", "v17",
                  "v18", "v19", "v20", "v21", "v22", "v23", "v27", "v28", "v29",
                  "v30"
#endif  // BNN_TYPE
            );

            bottom_value_0 += 2;
            bottom_value_1 += 2;
            bottom_value_2 += 2;
            bottom_value_3 += 2;

#else   // UNROLL_1x1_128
    FORZ(th, top_blob.h) {
        FORZ(tw, top_blob.w) {
            const auto *bottom_value_0 =
                bottom_blob.point<uint64_t>(th + 0, tw);

            // Replace this for loop by assembly
            FORZS(tc, weight.n, 4) {
                const auto *w_value_0 = weight.point<uint64_t>(tc + 0, 0, 0);

                auto *top_0_0 = top_blob.point<float>(th, tw) + tc;
                asm volatile(
                    "ld1    {v29.2d}, [%0]      \n"
                    "ld1	{v30.2d}, [%1]       \n"
                    "ld1    {v15.2d, v16.2d, v17.2d, v18.2d}, [%2]      \n"
                    "eor	v0.16b, v30.16b, v15.16b    \n"
                    "eor	v1.16b, v30.16b, v16.16b    \n"
                    "eor	v2.16b, v30.16b, v17.16b    \n"
                    "eor	v3.16b, v30.16b, v18.16b    \n"
                    "cnt	v0.16b, v0.16b        \n"
                    "cnt	v1.16b, v1.16b        \n"
                    "cnt	v2.16b, v2.16b        \n"
                    "cnt	v3.16b, v3.16b        \n"
                    "addv	b0, v0.16b       \n"
                    "addv	b1, v1.16b       \n"
                    "addv	b2, v2.16b       \n"
                    "addv	b3, v3.16b       \n"
                    "ins    v15.s[0], v0.s[0]                 \n"
                    "ins    v15.s[1], v1.s[0]                 \n"
                    "ins    v15.s[2], v2.s[0]                 \n"
                    "ins    v15.s[3], v3.s[0]                 \n"
                    "ucvtf  v15.4s, v15.4s     \n"
                    "st1   {v15.4s}, [%0]      \n"
                    // "add    %0.4s, %0.4s, v15.4s        \n"
                    : "+r"(top_0_0)         // %0
                    : "r"(bottom_value_0),  // %1
                      "r"(w_value_0)        // %2
                    : "cc", "memory", "x9", "x10", "x11", "x12", "v0", "v1",
                      "v2", "v3", "v15", "v16", "v17", "v18", "v29", "v30");
            }
#endif  // UNROLL_1x1_128
        }
    }
}

inline void bnn::bconv_1x1_64(const Mat &bottom_blob, const Mat &weight,
                              Mat &top_blob) {
    FORZ(th, top_blob.h) {
        FORZS(tw, top_blob.w, 1) {
            const auto *bottom_value_0 =
                bottom_blob.point<uint64_t>(th, tw + 0);

            const auto *w_value_0 = static_cast<uint64_t *>(weight.data);
            auto *top_0 = top_blob.point<float>(th, tw);
            size_t nn = weight.n >> 2;
            // TODO: optimize it
            asm volatile(
                "ld1	{v30.1d}, [%1]       \n"
                "0: \n"
                "prfm   pldl1keep, [%0, #128]     \n"
                "prfm   pldl1keep, [%2, #128]     \n"
                "ld1    {v13.1d, v14.1d, v15.1d, v16.1d}, [%2], #32      \n"
                "eor	v0.8b, v30.8b, v13.8b    \n"
                "eor	v1.8b, v30.8b, v14.8b    \n"
                "eor	v2.8b, v30.8b, v15.8b    \n"
                "eor	v3.8b, v30.8b, v16.8b    \n"
                "cnt	v0.8b, v0.8b        \n"
                "cnt	v1.8b, v1.8b        \n"
                "cnt	v2.8b, v2.8b        \n"
                "cnt	v3.8b, v3.8b        \n"
                "addv	b0, v0.8b       \n"
                "addv	b1, v1.8b       \n"
                "addv	b2, v2.8b       \n"
                "addv	b3, v3.8b       \n"
                "subs   %3, %3, #1           \n"
                "ins    v15.s[0], v0.s[0]                 \n"
                "ins    v15.s[1], v1.s[0]                 \n"
                "ins    v15.s[2], v2.s[0]                 \n"
                "ins    v15.s[3], v3.s[0]                 \n"
                "ucvtf  v15.4s, v15.4s     \n"
                "st1    {v15.4s}, [%0], #16        \n"
                "bne    0b              \n"
                : "+r"(top_0),           // %0
                  "+r"(bottom_value_0),  // %1
                  "+r"(w_value_0),       // %2
                  "+r"(nn)               // %3
                :
                : "cc", "memory", "v0", "v1", "v2", "v3", "v13", "v14", "v15",
                  "v16", "v30");
        }
    }
}
#endif // __aarch64__

#endif

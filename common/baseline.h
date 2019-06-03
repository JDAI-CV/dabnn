// Copyright 2019 JD.com Inc. JD AI

#ifndef BASELINE_H
#define BASELINE_H

#include <bitset>

#include <common/helper.h>
#include <dabnn/bitpack.h>
#include <dabnn/mat.h>

inline int bitcount(uint64_t x) {
#ifdef __aarch64__
    return __builtin_popcountl(x);
#else
    std::bitset<64> bs(x);
    return bs.count();
#endif
}

inline void baseline_pack_mat(const bnn::Mat &float_mat, bnn::Mat &binary_mat) {
    BNN_ASSERT(float_mat.c / 64 == binary_mat.c && float_mat.c % 64 == 0, "");
    FORZ(n, binary_mat.n) {
        FORZ(h, binary_mat.h) {
            FORZ(w, binary_mat.w) {
                FORZ(c, binary_mat.c) {
                    uint64_t *bptr = binary_mat.point<uint64_t>(n, h, w) + c;
                    pack_64_bitfield(float_mat.point<float>(n, h, w) + c * 64,
                                     bptr);
                }
            }
        }
    }
}

namespace bnn {
inline void baseline_fconv(const Mat &input, const Mat &weight,
                           const int kernel_h, const int kernel_w,
                           const int pad_h, const int pad_w, const int stride_h,
                           const int stride_w, const int dilation_h,
                           const int dilation_w, const int output_channels,
                           Mat &output) {
    auto top_ptr = static_cast<float *>(output);
    int input_y = 0;
    FORZ(th, output.h) {
        int input_x = 0;
        FORZ(tw, output.w) {
            FORZ(tc, output_channels) {
                FORZ(wh, kernel_h) {
                    int y = input_y - pad_h + wh * dilation_h;
                    FORZ(ww, kernel_w) {
                        int x = input_x - pad_w + ww * dilation_w;
                        FORZ(wc, input.c) {
                            int idx = tc * kernel_h * kernel_w * input.c +
                                      wh * kernel_w * input.c + ww * input.c +
                                      wc;
                            const auto w_value =
                                weight[idx];  // weight.point<float>(tc,
                                              // wh, ww) + wc;
                            bool out =
                                y < 0 || y >= input.h || x < 0 || x >= input.w;
                            const auto bottom_value =
                                out ? 0 : *(input.point<float>(y, x) + wc);
                            float tmp = (w_value) * (bottom_value);
                            top_ptr[th * output.w * output.c + tw * output.c +
                                    tc] += tmp;
                        }
                    }
                }
            }
            input_x += stride_w;
        }
        input_y += stride_h;
    }
}

inline void baseline_bconv(const Mat &input, const Mat &weight,
                           const int kernel_h, const int kernel_w,
                           const int pad_h, const int pad_w, const int stride_h,
                           const int stride_w, const int dilation_h,
                           const int dilation_w, const int output_channels,
                           Mat &output) {
    int input_y = 0;
    FORZ(th, output.h) {
        int input_x = 0;
        FORZ(tw, output.w) {
            FORZ(tc, output_channels) {
                uint32_t acc = 0;
                FORZ(wh, kernel_h) {
                    int y = input_y - pad_h + wh * dilation_h;
                    FORZ(ww, kernel_w) {
                        int x = input_x - pad_w + ww * dilation_w;
                        FORZ(wc, input.c) {
                            int idx = tc * kernel_h * kernel_w * input.c +
                                      wh * kernel_w * input.c + ww * input.c +
                                      wc;
                            const auto w_value =
                                *(static_cast<uint64_t *>(weight.data) + idx);
                            bool out =
                                y < 0 || y >= input.h || x < 0 || x >= input.w;
                            const auto bottom_value =
                                out ? 0 : *(input.point<uint64_t>(y, x) + wc);
                            uint8_t tmp = ::bitcount(w_value ^ bottom_value);
                            acc += tmp;
                        }
                    }
                }
                *(output.point<float>(th, tw) + tc) = static_cast<float>(acc);
            }
            input_x += stride_w;
        }
        input_y += stride_h;
    }
}

inline void baseline_bconv_float(const Mat &input, Mat &binary_input,
                                 const Mat &weight, const int kernel_h,
                                 const int kernel_w, const int pad_h,
                                 const int pad_w, const int stride_h,
                                 const int stride_w, const int dilation_h,
                                 const int dilation_w,
                                 const int output_channels, Mat &output) {
    pack_mat_64(input, binary_input);

    baseline_bconv(binary_input, weight, kernel_h, kernel_w, pad_h, pad_w,
                   stride_h, stride_w, dilation_h, dilation_w, output_channels,
                   output);
}
}  // namespace bnn

#endif /* BASELINE_H */

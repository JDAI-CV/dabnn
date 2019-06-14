#include <dabnn/bitpack.h>
#include <dabnn/im2col.h>
#include <dabnn/mat.h>

namespace bnn {
inline void fused_binarize_im2col(const Mat &im, const int kernel_h,
                                  const int kernel_w, const int pad_h,
                                  const int pad_w, const int stride_h,
                                  const int stride_w, const int dilation_h,
                                  const int dilation_w, Mat &col) {
    const int output_h =
        (im.h + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    const int output_w =
        (im.w + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

    BNN_ASSERT(kernel_h * kernel_w * im.c < 60000,
               "kernel_h * kernel_w * im.c must be smaller than 60000");
    // TODO: More elegant way
    static float buf[60000];

    char *data_col = static_cast<char *>(col);
    int input_y = 0;
    FORZ(output_y, output_h) {
        int input_x = 0;
        FORZ(output_x, output_w) {
            float *buf_ptr = buf;
            FORZ(kh, kernel_h) {
                int y = input_y - pad_h + kh * dilation_h;
                const char *data_im = static_cast<char *>(im.data) +
                                      y * im.w * im.c * im.elemsize;
                FORZ(kw, kernel_w) {
                    int x = input_x - pad_w + kw * dilation_w;
                    if (y < 0 || y >= im.h || x < 0 || x >= im.w) {
                        memset(buf_ptr, 0, im.c * im.elemsize);
                    } else {
                        memcpy(buf_ptr, data_im + x * im.c * im.elemsize,
                               im.c * im.elemsize);
                    }
                    buf_ptr += im.c * im.elemsize;
                }
            }

            const size_t len = (buf_ptr - buf) / im.elemsize;
            const size_t len_aligned_128 = (len + 127) / 128 * 128;
            // pad the buffer so that its length aligns to 128
            memset(buf_ptr, 0, (len_aligned_128 - len) * im.elemsize);

            pack_128_opt(buf_ptr, data_col, len_aligned_128);

            // `len_aligned_128` is the number of appended __bits__ in
            // mat `col`, so divide sizeof(decltype(data_col)) here
            data_col += len_aligned_128 / sizeof(decltype(data_col));

            input_x += stride_w;
        }
        input_y += stride_h;
    }
}
}  // namespace bnn

// Copyright 2019 JD.com Inc. JD AI

#include "Affine.h"

namespace bnn {

/**
 * per channel affine, x = a * x + b
 */
inline void affine_inplace(bnn::Mat &data, const bnn::Mat &a,
                           const bnn::Mat &b) {
    FORZ(n, data.n) {
        FORZ(h, data.h) {
            auto ptr = data.point<float>(n, h, 0);
            FORZ(w, data.w) {
                FORZ(c, data.c) {
                    *ptr = a[c] * *ptr + b[c];
                    ptr++;
                }
            }
        }
    }
}

inline void affine(const bnn::Mat &data, const bnn::Mat &a,
                           const bnn::Mat &b, bnn::Mat &output) {
    FORZ(n, data.n) {
        FORZ(h, data.h) {
            const auto *ptr = data.point<float>(n, h, 0);
            auto output_ptr = output.point<float>(n, h, 0);
            FORZ(w, data.w) {
                FORZ(c, data.c) {
                    *output_ptr = a[c] * *ptr + b[c];
                    ptr++;
                    output_ptr++;
                }
            }
        }
    }
}

void Affine::forward_impl() const {
#ifdef BNN_CHECK_CONSISTENCY
    affine(*data_mat, *a_mat, *b_mat, *output_mat);
#else
    affine_inplace(*data_mat, *a_mat, *b_mat); 
#endif
}

}  // namespace bnn

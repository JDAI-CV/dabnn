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

void Affine::forward_impl() const { affine_inplace(*data_mat, *a_mat, *b_mat); }

}  // namespace bnn

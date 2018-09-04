// Copyright 2019 JD.com Inc. JD AI

#include "Concat.h"

namespace bnn {

void Concat::forward_impl() const {
    const auto *in_ptr1 = static_cast<float *>(*input1_mat);
    const auto *in_ptr2 = static_cast<float *>(*input2_mat);
    auto *out_ptr = static_cast<float *>(*output_mat);

    FORZ(h, input1_mat->h) {
        FORZ(w, input1_mat->w) {
            memcpy(out_ptr, in_ptr1, input1_mat->c * sizeof(float));
            out_ptr += input1_mat->c;
            in_ptr1 += input1_mat->c;
            memcpy(out_ptr, in_ptr2, input2_mat->c * sizeof(float));
            out_ptr += input2_mat->c;
            in_ptr2 += input2_mat->c;
        }
    }
}

}  // namespace bnn

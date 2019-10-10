// Copyright 2019 JD.com Inc. JD AI

#include "PRelu.h"

namespace bnn {
void PRelu::forward_impl() const {
    BNN_ASSERT(slope_mat->total() == 1 ||
                   slope_mat->total() == static_cast<size_t>(data_mat->c),
               "slope must have size 1 or input.channels");
    float *ptr = static_cast<float *>(*data_mat);
    float *slope_ptr = static_cast<float *>(*slope_mat);
    if (slope_mat->total() == 1) {
        const auto slope = *slope_ptr;
        FORZ(i, data_mat->total()) {
            if (*ptr < 0) {
                *ptr = (*ptr) * slope;
            }
            ptr++;
        }
    } else if (slope_mat->total() == static_cast<size_t>(data_mat->c)) {
        const auto nhw = data_mat->n * data_mat->h * data_mat->w;
        FORZ(i, nhw) {
            FORZ(j, data_mat->c) {
                if (*ptr < 0) {
                    *ptr = (*ptr) * slope_ptr[j];
                }
                ptr++;
            }
        }
    }
}
}  // namespace bnn

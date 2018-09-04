// Copyright 2019 JD.com Inc. JD AI

#include "Split.h"

namespace bnn {
void Split::forward_impl() const {
    const auto c = input_mat->c;
    const auto c_per_output = c / 2;
    auto *ptr = static_cast<uint64_t *>(input_mat->data);
    auto *ptr1 = static_cast<uint64_t *>(output_mat1->data);
    auto *ptr2 = static_cast<uint64_t *>(output_mat2->data);
    const auto nn = input_mat->total() / c;
    // PNT(nn);
    FORZ(_, nn) {
        memcpy(ptr1, ptr, c_per_output * sizeof(uint64_t));
        ptr1 += c_per_output;
        ptr += c_per_output;
        memcpy(ptr2, ptr, c_per_output * sizeof(uint64_t));
        ptr2 += c_per_output;
        ptr += c_per_output;
    }
}
}  // namespace bnn

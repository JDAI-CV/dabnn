// Copyright 2019 JD.com Inc. JD AI

#include "Shuffle.h"

namespace bnn {
void Shuffle::forward_impl() const {
    const auto c = data_mat->elem_c;
    const auto n = data_mat->total() / data_mat->c;
    if (c == 128) {
        VLOG(5) << "128 channels shuffle";
        auto *ptr = static_cast<uint32_t *>(data_mat->data);
        FORZ(_, n) {
            const auto tmp = *(ptr + 1);
            *(ptr + 1) = *(ptr + 2);
            *(ptr + 2) = tmp;
            ptr += 4;
        }
    } else if (c == 256) {
        VLOG(5) << "256 channels shuffle";
        auto *ptr = static_cast<uint64_t *>(data_mat->data);
        FORZ(_, n) {
            const auto tmp = *(ptr + 1);
            *(ptr + 1) = *(ptr + 2);
            *(ptr + 2) = tmp;
            ptr += 4;
        }
    } else if (c == 512) {
        VLOG(5) << "512 channels shuffle";
        auto *ptr = static_cast<uint64_t *>(data_mat->data);
        FORZ(_, n) {
            const auto tmp1 = *(ptr + 2);
            const auto tmp2 = *(ptr + 3);
            *(ptr + 2) = *(ptr + 4);
            *(ptr + 3) = *(ptr + 5);
            *(ptr + 4) = tmp1;
            *(ptr + 5) = tmp2;
            ptr += 8;
        }
    }
}
}  // namespace bnn

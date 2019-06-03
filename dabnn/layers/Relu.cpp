// Copyright 2019 JD.com Inc. JD AI

#include "Relu.h"

#if __ARM_NEON
#include <arm_neon.h>
#endif  // __ARM_NEON

namespace bnn {
void Relu::forward_impl() const {
#if __ARM_NEON
    float32x4_t _zero = vdupq_n_f32(0.f);
    float *ptr = static_cast<float *>(*data_mat);
    FORZ(i, data_mat->total() / 4) {
        float32x4_t _p = vld1q_f32(ptr);
        _p = vmaxq_f32(_p, _zero);
        vst1q_f32(ptr, _p);

        ptr += 4;
    }
#else
    float *ptr = static_cast<float *>(*data_mat);
    FORZ(i, data_mat->total()) {
        *ptr = std::max(*ptr, 0.f);
    }
#endif // __ARM_NEON
}
}  // namespace bnn

// Copyright 2019 JD.com Inc. JD AI
//
// The step of bit-packing packs N 32-bit float/integer to an N-bit
// operand according their signs. For example, performing bit-packing
// on 128 float numbers produces a 128-bit operand. xnor/xor is only
// enabled on these packed operands.
//
// The method in this file is usually for the packing of input. The
// packing of weight has been performed offline in the step of
// onnx2bnn.

#ifndef BITPACK_H
#define BITPACK_H

#if __ARM_NEON
#include <arm_neon.h>
#endif  // __ARM_NEON
#include <bitset>
#include <climits>
#include <cstdint>
#include <iostream>

#include <common/common_bitpack.h>
#include <common/helper.h>
#include <glog/logging.h>
#include "mat.h"

namespace bnn {

#ifdef __aarch64__
inline void pack_128_opt(const float *float_ptr, void *binary_ptr,
                         size_t size) {
    /**
     * size: the number of __elements__ needed to be packed.
     *
     * This is the optimized bit-packing.
     *
     * sri is the "shift-right-and-overwrite" instruction.
     * By this instruction, we directly leveraging the existing
     * sign bits in 32-bit operands (both IEEE 754 float and
     * 32-bit integer).
     * Note that the order of bits in the output operand is not
     * the consistent with the order of input operands. Fortunately,
     * this consistency is not indispensable -- the result of
     * xnor/xor is still correct as long as the bits of both input
     * and weight are re-arranged in the same way.
     * Therefore, we re-arrange the packed weight accordingly in
     * dabnn/net.cpp
     */
    size_t nn_size = size >> 7;

    asm volatile(
        "0:     \n"
        "prfm   pldl1keep, [%0]     \n"
        "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%0], #64    \n"
        "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%0], #64    \n"
        "sri    v0.4s, v4.4s, #1    \n"
        "sri    v1.4s, v5.4s, #1    \n"
        "sri    v2.4s, v6.4s, #1    \n"
        "sri    v3.4s, v7.4s, #1    \n"

        "ld1    {v8.4s, v9.4s, v10.4s, v11.4s}, [%0], #64    \n"
        "prfm   pldl1keep, [%0, #64]     \n"
        "ld1    {v12.4s, v13.4s, v14.4s, v15.4s}, [%0], #64    \n"
        "sri    v8.4s, v12.4s, #1    \n"
        "sri    v9.4s, v13.4s, #1    \n"
        "sri    v10.4s, v14.4s, #1    \n"
        "sri    v11.4s, v15.4s, #1    \n"

        "subs   %2, %2, #1          \n"

        "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%0], #64    \n"
        "prfm   pldl1keep, [%0, #64]     \n"
        "ld1    {v20.4s, v21.4s, v22.4s, v23.4s}, [%0], #64    \n"

        "sri    v0.4s, v8.4s, #2    \n"
        "sri    v1.4s, v9.4s, #2    \n"
        "sri    v2.4s, v10.4s, #2   \n"
        "sri    v3.4s, v11.4s, #2   \n"

        "sri    v16.4s, v20.4s, #1    \n"
        "sri    v17.4s, v21.4s, #1    \n"
        "sri    v18.4s, v22.4s, #1    \n"
        "sri    v19.4s, v23.4s, #1    \n"

        "ld1    {v8.4s, v9.4s, v10.4s, v11.4s}, [%0], #64    \n"
        "prfm   pldl1keep, [%0, #64]     \n"
        "ld1    {v12.4s, v13.4s, v14.4s, v15.4s}, [%0], #64    \n"
        "sri    v8.4s, v12.4s, #1    \n"
        "sri    v9.4s, v13.4s, #1    \n"
        "sri    v10.4s, v14.4s, #1    \n"
        "sri    v11.4s, v15.4s, #1    \n"

        "sri    v16.4s, v8.4s, #2   \n"
        "sri    v17.4s, v9.4s, #2   \n"
        "sri    v18.4s, v10.4s, #2   \n"
        "sri    v19.4s, v11.4s, #2   \n"

        "sri    v0.4s, v16.4s, #4   \n"
        "sri    v1.4s, v17.4s, #4   \n"
        "sri    v2.4s, v18.4s, #4   \n"
        "sri    v3.4s, v19.4s, #4   \n"

        "sri    v0.4s, v1.4s, #8    \n"
        "sri    v2.4s, v3.4s, #8    \n"
        "sri    v0.4s, v2.4s, #16    \n"

        // Bit-packing with sign bit is introduced after the first version
        // of dabnn is published. Sign bit will be 1 when x < 0, 0 when x > 0,
        // which is different with the way we used before --- set bit to 1 if
        // x > 0 or 0 if x < 0
        // So for the compatibility we add a "not" instruction here.
        // Maybe we can save this instruction by introducing "version" for
        // dabnn model and force users to upgrade.
        // Note: If this line is removed, the padding value of binary
        // convolution should also be changed from 0 (-1 in xnor) to -1 (1 in
        // xnor)
        "not    v0.16b, v0.16b        \n"

        "st1    {v0.4s}, [%1], #16         \n"
        "bne    0b                  \n"
        : "+r"(float_ptr),   // %0
          "+r"(binary_ptr),  // %1
          "+r"(nn_size)      // %2
        :
        : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8",
          "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18",
          "v19", "v20", "v21", "v22", "v23", "x0");
}

inline void pack_128_baseline(const float *float_ptr, void *binary_ptr,
                              size_t size) {
    size_t nn_size = size >> 7;

    asm volatile(
        "0:     \n"
        "prfm   pldl1keep, [%0]     \n"
        "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%0], #64    \n"
        "cmge   v0.4s, v0.4s, #0    \n"
        "cmge   v1.4s, v1.4s, #0    \n"
        "cmge   v2.4s, v2.4s, #0    \n"
        "sli    v0.4s, v1.4s, #1    \n"
        "cmge   v3.4s, v3.4s, #0    \n"
        "sli    v0.4s, v2.4s, #2    \n"
        "ld1    {v5.4s, v6.4s, v7.4s, v8.4s}, [%0], #64    \n"
        "sli    v0.4s, v3.4s, #3    \n"
        "cmge   v5.4s, v5.4s, #0    \n"
        "cmge   v6.4s, v6.4s, #0    \n"
        "sli    v0.4s, v5.4s, #4    \n"
        "cmge   v7.4s, v7.4s, #0    \n"
        "sli    v0.4s, v6.4s, #5    \n"
        "cmge   v8.4s, v8.4s, #0    \n"
        "sli    v0.4s, v7.4s, #6    \n"
        "ld1    {v1.4s, v2.4s, v3.4s, v4.4s}, [%0], #64    \n"
        "cmge   v1.4s, v1.4s, #0    \n"
        "sli    v0.4s, v8.4s, #7    \n"
        "cmge   v2.4s, v2.4s, #0    \n"
        "sli    v0.4s, v1.4s, #8    \n"
        "cmge   v3.4s, v3.4s, #0    \n"
        "sli    v0.4s, v2.4s, #9    \n"
        "cmge   v4.4s, v4.4s, #0    \n"
        "sli    v0.4s, v3.4s, #10   \n"
        "ld1    {v5.4s, v6.4s, v7.4s, v8.4s}, [%0], #64    \n"
        "sli    v0.4s, v4.4s, #11   \n"
        "cmge   v5.4s, v5.4s, #0    \n"
        "ld1    {v1.4s, v2.4s, v3.4s, v4.4s}, [%0], #64    \n"
        "cmge   v6.4s, v6.4s, #0    \n"
        "sli    v0.4s, v5.4s, #12   \n"
        "cmge   v7.4s, v7.4s, #0    \n"
        "sli    v0.4s, v6.4s, #13   \n"
        "cmge   v8.4s, v8.4s, #0    \n"
        "sli    v0.4s, v7.4s, #14   \n"
        "cmge   v1.4s, v1.4s, #0    \n"
        "sli    v0.4s, v8.4s, #15   \n"
        "cmge   v2.4s, v2.4s, #0    \n"
        "sli    v0.4s, v1.4s, #16   \n"
        "cmge   v3.4s, v3.4s, #0    \n"
        "sli    v0.4s, v2.4s, #17   \n"
        "cmge   v4.4s, v4.4s, #0    \n"
        "sli    v0.4s, v3.4s, #18   \n"
        "ld1    {v5.4s, v6.4s, v7.4s, v8.4s}, [%0], #64    \n"
        "sli    v0.4s, v4.4s, #19   \n"
        "cmge   v5.4s, v5.4s, #0    \n"
        "ld1    {v1.4s, v2.4s, v3.4s, v4.4s}, [%0], #64    \n"
        "cmge   v6.4s, v6.4s, #0    \n"
        "sli    v0.4s, v5.4s, #20   \n"
        "cmge   v7.4s, v7.4s, #0    \n"
        "sli    v0.4s, v6.4s, #21   \n"
        "cmge   v8.4s, v8.4s, #0    \n"
        "sli    v0.4s, v7.4s, #22   \n"
        "cmge   v1.4s, v1.4s, #0    \n"
        "sli    v0.4s, v8.4s, #23   \n"
        "cmge   v2.4s, v2.4s, #0    \n"
        "sli    v0.4s, v1.4s, #24   \n"
        "cmge   v3.4s, v3.4s, #0    \n"
        "sli    v0.4s, v2.4s, #25   \n"
        "cmge   v4.4s, v4.4s, #0    \n"
        "sli    v0.4s, v3.4s, #26   \n"
        "subs   %2, %2, #1          \n"
        "ld1    {v5.4s, v6.4s, v7.4s, v8.4s}, [%0], #64    \n"
        "sli    v0.4s, v4.4s, #27   \n"
        "cmge   v5.4s, v5.4s, #0    \n"
        "cmge   v6.4s, v6.4s, #0    \n"
        "sli    v0.4s, v5.4s, #28   \n"
        "cmge   v7.4s, v7.4s, #0    \n"
        "sli    v0.4s, v6.4s, #29   \n"
        "cmge   v8.4s, v8.4s, #0    \n"
        "sli    v0.4s, v7.4s, #30   \n"
        "sli    v0.4s, v8.4s, #31   \n"
        "st1    {v0.4s}, [%1], #16         \n"
        "bne    0b                  \n"
        : "+r"(float_ptr),   // %0
          "+r"(binary_ptr),  // %1
          "+r"(nn_size)      // %2
        :
        : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8",
          "x0");
}

inline void pack_mat_128_opt(const bnn::Mat &float_mat, bnn::Mat &binary_mat) {
    BNN_ASSERT(!binary_mat.empty(), "binary_mat must not be empty");

    pack_128_opt(static_cast<float *>(float_mat.data), binary_mat.data,
                 float_mat.total());
}

inline void pack_mat_128_baseline(const bnn::Mat &float_mat,
                                  bnn::Mat &binary_mat) {
    BNN_ASSERT(!binary_mat.empty(), "binary_mat must not be empty");

    pack_128_baseline(static_cast<float *>(float_mat.data), binary_mat.data,
                      float_mat.total());
}

inline void pack_mat_128(const bnn::Mat &float_mat, bnn::Mat &binary_mat) {
    /**
     * Delegate it to optimized implementation.
     * The cost of function calling will be eliminated by compiler,
     * don't bother.
     */
    pack_mat_128_opt(float_mat, binary_mat);
}
#endif  // __aarch64__

inline void pack_mat_64(const bnn::Mat &float_mat, bnn::Mat &binary_mat) {
    /**
     * This is the bit-packing for tensor of less than 128 channels.
     */
    BNN_ASSERT(
        float_mat.w * float_mat.c > 0 && float_mat.w * float_mat.c % 64 == 0,
        float_mat.w * float_mat.c);
    BNN_ASSERT(float_mat.c / 64 == binary_mat.c && float_mat.c % 64 == 0, "");

    FORZ(n, float_mat.n) {
        FORZ(h, float_mat.h) {
            auto *fptr = float_mat.point<float>(n, h, 0);
            auto *bptr = binary_mat.point<uint64_t>(n, h, 0);
            FORZ(i, float_mat.w * float_mat.c / 64) {
                pack_64_bitfield(fptr, bptr);
                fptr += 64;
                bptr++;
            }
        }
    }
}

inline void pack_mat(const bnn::Mat &float_mat, bnn::Mat &binary_mat) {
    BNN_ASSERT(float_mat.c % 64 == 0, float_mat.c);
#ifdef __aarch64__
    if (float_mat.c % 128 == 0) {
        pack_mat_128_opt(float_mat, binary_mat);
    } else {
        pack_mat_64(float_mat, binary_mat);
    }
#else
    pack_mat_64(float_mat, binary_mat);
#endif  // __aarch64__
}

}
#endif /* BITPACK_H */

// Copyright 2019 JD.com Inc. JD AI

#ifndef BNN_BGEMM_H
#define BNN_BGEMM_H

#if __ARM_NEON
#include <arm_neon.h>
#endif  // __ARM_NEON
#include <common/baseline.h>
#include <common/helper.h>

#if __ARM_NEON
#ifdef __aarch64__
#define P 8
#define R 6
#else
#define P 4
#define R 4
#endif // __aarch64__
#endif // __ARM_NEON

#define A(i, j) a[(j)*lda + (i)]  // A(y, x)
#define B(i, j) b[(j)*ldb + (i)]  // B(y, x)
#define C(i, j) c[(j)*ldc + (i)]  // C(y, x)

#define min(i, j) ((i) < (j) ? (i) : (j))

#ifdef __ARM_NEON
inline void pack_a(const int kc, const uint64_t *a, const int lda,
                   uint64_t *a_to);
inline void pack_b(const int kc, const uint64_t *b, const int ldb,
                   uint64_t *b_to);
inline void unpack_c(const float *c_from, const int ldc, float *c,
                     const int block_row, const int block_col);
inline void micro_kernel(int64_t kc, float *c, const uint64_t *a,
                         const uint64_t *b);
inline void inner_kernel(const int m, const int n, const int k,
                         const uint64_t *a, const int lda, const uint64_t *b,
                         const int ldb, float *c, const int ldc,
                         const int first_time);
#endif  // __ARM_NEON
inline void bgemm_naive(const int m, const int n, const int k,
                        const uint64_t *a, const int lda, const uint64_t *b,
                        const int ldb, float *c, const int ldc);

inline void bgemm(const int m, const int n, const int k, const uint64_t *a,
                  int lda, const uint64_t *b, const int ldb, float *c,
                  const int ldc) {
#ifdef __ARM_NEON
    int kc = 32;
    int mc = 32;
    int i, q, qb, ib;

    for (q = 0; q < k; q += kc) {
        qb = min(k - q, kc);

        for (i = 0; i < m; i += mc) {
            ib = min(m - i, mc);
            inner_kernel(ib, n, qb, &A(i, q), lda, &B(q, 0), ldb, &C(i, 0), ldc,
                         i == 0);
        }
    }
#else
    bgemm_naive(m, n, k, a, lda, b, ldb, c, ldc);
#endif  // __ARM_NEON
}

#if __ARM_NEON
inline void inner_kernel(const int m, const int n, const int k,
                         const uint64_t *a, const int lda, const uint64_t *b,
                         const int ldb, float *c, const int ldc,
                         const int first_time) {
    BNN_ASSERT(k % 2 == 0, "k % 2 should be 0");
    BNN_ASSERT(k * P < 128000, "");
    BNN_ASSERT(k * R < 128000, "");

    int i = 0, j = 0;
    // TODO: more elegant way
    alignas(128) static uint64_t packedA[128000];
    alignas(128) static uint64_t packedB[128000];
    alignas(128) static float packedC[P * R];

    for (j = 0; j + R <= n; j += R) {
        if (first_time) pack_b(k, &B(0, j), ldb, &packedB[j * k]);
        for (i = 0; i + P <= m; i += P) {
            if (j == 0) pack_a(k, &A(i, 0), lda, &packedA[i * k]);
            memset(packedC, 0, P * R * 4);
            // k/2: k is the amount of uint64_t, k/2 is the amount of 128bit
            // vector
            micro_kernel(k / 2, packedC, &packedA[i * k], &packedB[j * k]);
            unpack_c(packedC, ldc, &C(i, j), 0, 0);
        }
    }
    if (i != m) {
        FOR(_j, 0, j) {
            FOR(_i, i, m) {
                FORZ(_k, k) { C(_i, _j) += bitcount(A(_i, _k) ^ B(_k, _j)); }
            }
        }
    }
    if (j != n) {
        FOR(_j, j, n) {
            FOR(_i, 0, i) {
                FORZ(_k, k) { C(_i, _j) += bitcount(A(_i, _k) ^ B(_k, _j)); }
            }
        }
    }
    if (i != m || j != n) {
        FOR(_j, j, n) {
            FOR(_i, i, m) {
                FORZ(_k, k) { C(_i, _j) += bitcount(A(_i, _k) ^ B(_k, _j)); }
            }
        }
    }
}

inline void pack_a(const int kc, const uint64_t *a, const int lda,
                   uint64_t *a_to) {
    for (int i = 0; i < kc; i += 2) {
        for (int j = 0; j < P; j++) {
            *a_to++ = A(j, i + 0);
            *a_to++ = A(j, i + 1);
        }
    }
}

inline void pack_b(const int kc, const uint64_t *b, const int ldb,
                   uint64_t *b_to) {
    for (int i = 0; i < kc; i += 2) {
        for (int j = 0; j < R; j++) {
            *b_to++ = B(i + 0, j);
            *b_to++ = B(i + 1, j);
        }
    }
}

inline void unpack_c(const float *c_from, const int ldc, float *c,
                     const int block_row, const int block_col) {
    for (int j = 0; j < R; j++) {
        for (int i = 0; i < P; i++) {
            C(block_row + i, block_col + j) += *c_from++;
        }
    }
}

inline void micro_kernel(int64_t kc, float *c, const uint64_t *a,
                         const uint64_t *b) {
#ifdef __aarch64__
    // C: 8x6(float 32, 6x2=12regs), A: 8*K(8regs), B: K*6(6regs)
    // v0~v11 contains C, v12~v17 contains 6*128 of B, v18~v25 contains 128*8 of
    // A v26~v30 store temporary values A is packed as
    // 8*128
    // -----
    // 8*128
    // B is packed as 128*6 | 128*6
    asm volatile(
        "mov x0, %1     \n"
        "ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x0], #64    \n"
        "ld1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x0], #64    \n"
        "ld1 {v8.4s, v9.4s, v10.4s, v11.4s}, [x0]    \n"

        "0: \n"
        "ld1 {v18.2d, v19.2d, v20.2d, v21.2d}, [%3], #64    \n"
        "ld1 {v12.2d, v13.2d, v14.2d, v15.2d}, [%2], #64    \n"
        "eor v26.16b, v12.16b, v18.16b         \n"
        "eor v27.16b, v12.16b, v19.16b         \n"
        "eor v28.16b, v12.16b, v20.16b         \n"
        "eor v29.16b, v12.16b, v21.16b         \n"
        "cnt v26.16b, v26.16b                 \n"
        "cnt v27.16b, v27.16b                 \n"
        "cnt v28.16b, v28.16b                 \n"
        "cnt v29.16b, v29.16b                 \n"
        "ld1 {v22.2d, v23.2d, v24.2d, v25.2d}, [%3], #64    \n"
        "addv b26, v26.16b                \n"
        "addv b27, v27.16b                \n"
        "addv b28, v28.16b                \n"
        "addv b29, v29.16b                \n"
        "ins v26.s[1], v27.s[0]             \n"
        "ins v26.s[2], v28.s[0]             \n"
        "ins v26.s[3], v29.s[0]             \n"

        "ld1 {v16.2d, v17.2d}, [%2], #32    \n"
        "eor v27.16b, v12.16b, v22.16b         \n"
        "eor v28.16b, v12.16b, v23.16b         \n"
        "eor v29.16b, v12.16b, v24.16b         \n"
        "eor v30.16b, v12.16b, v25.16b         \n"
        "cnt v27.16b, v27.16b                 \n"
        "cnt v28.16b, v28.16b                 \n"
        "cnt v29.16b, v29.16b                 \n"
        "cnt v30.16b, v30.16b                 \n"
        "addv b27, v27.16b                \n"
        "addv b28, v28.16b                \n"
        "addv b29, v29.16b                \n"
        "addv b30, v30.16b                \n"
        "ins v27.s[1], v28.s[0]             \n"
        "ins v27.s[2], v29.s[0]             \n"
        "ins v27.s[3], v30.s[0]             \n"

        "add v0.4s, v0.4s, v26.4s           \n"  // delay
        "add v1.4s, v1.4s, v27.4s           \n"

        // The No.1 col of C is finished

        "eor v26.16b, v13.16b, v18.16b         \n"
        "eor v27.16b, v13.16b, v19.16b         \n"
        "eor v28.16b, v13.16b, v20.16b         \n"
        "eor v29.16b, v13.16b, v21.16b         \n"
        "cnt v26.16b, v26.16b                 \n"
        "cnt v27.16b, v27.16b                 \n"
        "cnt v28.16b, v28.16b                 \n"
        "cnt v29.16b, v29.16b                 \n"
        "addv b26, v26.16b                \n"
        "addv b27, v27.16b                \n"
        "addv b28, v28.16b                \n"
        "addv b29, v29.16b                \n"
        "ins v26.s[1], v27.s[0]             \n"
        "ins v26.s[2], v28.s[0]             \n"
        "ins v26.s[3], v29.s[0]             \n"

        "eor v27.16b, v13.16b, v22.16b         \n"
        "eor v28.16b, v13.16b, v23.16b         \n"
        "eor v29.16b, v13.16b, v24.16b         \n"
        "eor v30.16b, v13.16b, v25.16b         \n"
        "cnt v27.16b, v27.16b                 \n"
        "cnt v28.16b, v28.16b                 \n"
        "cnt v29.16b, v29.16b                 \n"
        "cnt v30.16b, v30.16b                 \n"
        "addv b27, v27.16b                \n"
        "addv b28, v28.16b                \n"
        "addv b29, v29.16b                \n"
        "addv b30, v30.16b                \n"
        "ins v27.s[1], v28.s[0]             \n"
        "ins v27.s[2], v29.s[0]             \n"
        "ins v27.s[3], v30.s[0]             \n"

        "add v2.4s, v2.4s, v26.4s           \n"  // delay
        "add v3.4s, v3.4s, v27.4s           \n"

        // The No.2 col of C is finished

        "eor v26.16b, v14.16b, v18.16b         \n"
        "eor v27.16b, v14.16b, v19.16b         \n"
        "eor v28.16b, v14.16b, v20.16b         \n"
        "eor v29.16b, v14.16b, v21.16b         \n"
        "cnt v26.16b, v26.16b                 \n"
        "cnt v27.16b, v27.16b                 \n"
        "cnt v28.16b, v28.16b                 \n"
        "cnt v29.16b, v29.16b                 \n"
        "addv b26, v26.16b                \n"
        "addv b27, v27.16b                \n"
        "addv b28, v28.16b                \n"
        "addv b29, v29.16b                \n"
        "ins v26.s[1], v27.s[0]             \n"
        "ins v26.s[2], v28.s[0]             \n"
        "ins v26.s[3], v29.s[0]             \n"

        "eor v27.16b, v14.16b, v22.16b         \n"
        "eor v28.16b, v14.16b, v23.16b         \n"
        "eor v29.16b, v14.16b, v24.16b         \n"
        "eor v30.16b, v14.16b, v25.16b         \n"
        "cnt v27.16b, v27.16b                 \n"
        "cnt v28.16b, v28.16b                 \n"
        "cnt v29.16b, v29.16b                 \n"
        "cnt v30.16b, v30.16b                 \n"
        "addv b27, v27.16b                \n"
        "addv b28, v28.16b                \n"
        "addv b29, v29.16b                \n"
        "addv b30, v30.16b                \n"
        "ins v27.s[1], v28.s[0]             \n"
        "ins v27.s[2], v29.s[0]             \n"
        "ins v27.s[3], v30.s[0]             \n"

        "add v4.4s, v4.4s, v26.4s           \n"  // delay
        "add v5.4s, v5.4s, v27.4s           \n"

        // The No.3 col of C is finished

        "prfm   pldl1keep, [%3, #128]     \n"
        "eor v26.16b, v15.16b, v18.16b         \n"
        "eor v27.16b, v15.16b, v19.16b         \n"
        "eor v28.16b, v15.16b, v20.16b         \n"
        "eor v29.16b, v15.16b, v21.16b         \n"
        "cnt v26.16b, v26.16b                 \n"
        "cnt v27.16b, v27.16b                 \n"
        "cnt v28.16b, v28.16b                 \n"
        "cnt v29.16b, v29.16b                 \n"
        "addv b26, v26.16b                \n"
        "addv b27, v27.16b                \n"
        "addv b28, v28.16b                \n"
        "addv b29, v29.16b                \n"
        "ins v26.s[1], v27.s[0]             \n"
        "ins v26.s[2], v28.s[0]             \n"
        "ins v26.s[3], v29.s[0]             \n"

        "prfm   pldl1keep, [%2, #128]     \n"
        "eor v27.16b, v15.16b, v22.16b         \n"
        "eor v28.16b, v15.16b, v23.16b         \n"
        "eor v29.16b, v15.16b, v24.16b         \n"
        "eor v30.16b, v15.16b, v25.16b         \n"
        "cnt v27.16b, v27.16b                 \n"
        "cnt v28.16b, v28.16b                 \n"
        "cnt v29.16b, v29.16b                 \n"
        "cnt v30.16b, v30.16b                 \n"
        "addv b27, v27.16b                \n"
        "addv b28, v28.16b                \n"
        "addv b29, v29.16b                \n"
        "addv b30, v30.16b                \n"
        "ins v27.s[1], v28.s[0]             \n"
        "ins v27.s[2], v29.s[0]             \n"
        "ins v27.s[3], v30.s[0]             \n"

        "add v6.4s, v6.4s, v26.4s           \n"  // delay
        "add v7.4s, v7.4s, v27.4s           \n"

        // The No.4 col of C is finished

        "eor v26.16b, v16.16b, v18.16b         \n"
        "eor v27.16b, v16.16b, v19.16b         \n"
        "eor v28.16b, v16.16b, v20.16b         \n"
        "eor v29.16b, v16.16b, v21.16b         \n"
        "cnt v26.16b, v26.16b                 \n"
        "cnt v27.16b, v27.16b                 \n"
        "cnt v28.16b, v28.16b                 \n"
        "cnt v29.16b, v29.16b                 \n"
        "addv b26, v26.16b                \n"
        "addv b27, v27.16b                \n"
        "addv b28, v28.16b                \n"
        "addv b29, v29.16b                \n"
        "ins v26.s[1], v27.s[0]             \n"
        "ins v26.s[2], v28.s[0]             \n"
        "ins v26.s[3], v29.s[0]             \n"

        "eor v27.16b, v16.16b, v22.16b         \n"
        "eor v28.16b, v16.16b, v23.16b         \n"
        "eor v29.16b, v16.16b, v24.16b         \n"
        "eor v30.16b, v16.16b, v25.16b         \n"
        "cnt v27.16b, v27.16b                 \n"
        "cnt v28.16b, v28.16b                 \n"
        "cnt v29.16b, v29.16b                 \n"
        "cnt v30.16b, v30.16b                 \n"
        "addv b27, v27.16b                \n"
        "addv b28, v28.16b                \n"
        "addv b29, v29.16b                \n"
        "addv b30, v30.16b                \n"
        "ins v27.s[1], v28.s[0]             \n"
        "ins v27.s[2], v29.s[0]             \n"
        "ins v27.s[3], v30.s[0]             \n"

        "add v8.4s, v8.4s, v26.4s           \n"  // delay
        "add v9.4s, v9.4s, v27.4s           \n"

        // The No.5 col of C is finished

        "eor v26.16b, v17.16b, v18.16b         \n"
        "eor v27.16b, v17.16b, v19.16b         \n"
        "eor v28.16b, v17.16b, v20.16b         \n"
        "eor v29.16b, v17.16b, v21.16b         \n"
        "cnt v26.16b, v26.16b                 \n"
        "cnt v27.16b, v27.16b                 \n"
        "cnt v28.16b, v28.16b                 \n"
        "cnt v29.16b, v29.16b                 \n"
        "addv b26, v26.16b                \n"
        "addv b27, v27.16b                \n"
        "addv b28, v28.16b                \n"
        "addv b29, v29.16b                \n"
        "ins v26.s[1], v27.s[0]             \n"
        "ins v26.s[2], v28.s[0]             \n"
        "ins v26.s[3], v29.s[0]             \n"

        "subs %0, %0, #1    \n"

        "eor v27.16b, v17.16b, v22.16b         \n"
        "eor v28.16b, v17.16b, v23.16b         \n"
        "eor v29.16b, v17.16b, v24.16b         \n"
        "eor v30.16b, v17.16b, v25.16b         \n"
        "cnt v27.16b, v27.16b                 \n"
        "cnt v28.16b, v28.16b                 \n"
        "cnt v29.16b, v29.16b                 \n"
        "cnt v30.16b, v30.16b                 \n"
        "addv b27, v27.16b                \n"
        "addv b28, v28.16b                \n"
        "addv b29, v29.16b                \n"
        "addv b30, v30.16b                \n"
        "ins v27.s[1], v28.s[0]             \n"
        "ins v27.s[2], v29.s[0]             \n"
        "ins v27.s[3], v30.s[0]             \n"

        "add v10.4s, v10.4s, v26.4s           \n"  // delay
        "add v11.4s, v11.4s, v27.4s           \n"

        // The No.6 col of C is finished

        "bne 0b     \n"

        "ucvtf  v0.4s, v0.4s    \n"
        "ucvtf  v1.4s, v1.4s    \n"
        "ucvtf  v2.4s, v2.4s    \n"
        "ucvtf  v3.4s, v3.4s    \n"
        "ucvtf  v4.4s, v4.4s    \n"
        "ucvtf  v5.4s, v5.4s    \n"
        "st1 {v0.4s}, [%1], #16     \n"
        "ucvtf  v6.4s, v6.4s    \n"
        "st1 {v1.4s}, [%1], #16     \n"
        "ucvtf  v7.4s, v7.4s    \n"
        "st1 {v2.4s}, [%1], #16     \n"
        "ucvtf  v8.4s, v8.4s    \n"
        "st1 {v3.4s}, [%1], #16     \n"
        "ucvtf  v9.4s, v9.4s    \n"
        "st1 {v4.4s}, [%1], #16     \n"
        "ucvtf  v10.4s, v10.4s    \n"
        "st1 {v5.4s}, [%1], #16     \n"
        "ucvtf  v11.4s, v11.4s    \n"
        "st1 {v6.4s}, [%1], #16     \n"
        "st1 {v7.4s}, [%1], #16     \n"
        "st1 {v8.4s}, [%1], #16     \n"
        "st1 {v9.4s}, [%1], #16     \n"
        "st1 {v10.4s}, [%1], #16     \n"
        "st1 {v11.4s}, [%1], #16     \n"
        : "+r"(kc),  // %0
          "+r"(c),   // %1
          "+r"(b),   // %2
          "+r"(a)    // %3
        :
        : "cc", "memory", "x0", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
          "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17",
          "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27",
          "v28", "v29", "v30");
#else  // __aarch64__

    // C: 4x4(float 32, 4x1=4), A: 4*K(4regs), B: K*4(4regs)
    // q0~q3 contains C, q4~q7 contains 4*128 of B, q8~q11 contains 128*4 of A
    // q12~q15 store temporary values
    //
    // A is packed as
    // 4*128
    // -----
    // 4*128
    // B is packed as 128*4 | 128*4
    asm volatile(
        "mov r0, %1                     \n"
        "vld1.8 {q0-q1}, [r0]!    \n"
        "vld1.8 {q2-q3}, [r0]!    \n"
        "0: \n"
        "vld1.8 {q8-q9}, [%3]!    \n"
        "vld1.8 {q4-q5}, [%2]!    \n"
        "vld1.8 {q10-q11}, [%3]!    \n"
        "veor.u8  q12, q4, q8           \n"
        "veor.u8  q13, q4, q9           \n"
        "vcnt.u8  q12, q12          \n"
        "vcnt.u8  q13, q13          \n"
        "vld1.8 {q6-q7}, [%2]!    \n"
        "veor.u8  q14, q4, q10           \n"
        "veor.u8  q15, q4, q11           \n"
        "vcnt.u8  q14, q14          \n"
        "vcnt.u8  q15, q15          \n"
        "vpaddl.u8 q12, q12         \n"
        "vpaddl.u8 q13, q13         \n"
        "vpaddl.u8 q14, q14         \n"
        "vpaddl.u8 q15, q15         \n"
        "vpaddl.u16 q12, q12            \n"
        "vpaddl.u16 q13, q13            \n"
        "vpaddl.u16 q14, q14            \n"
        "vpaddl.u16 q15, q15            \n"
        "vpaddl.u32 q12, q12            \n"
        "vpaddl.u32 q13, q13            \n"
        "vpaddl.u32 q14, q14            \n"
        "vpaddl.u32 q15, q15            \n"
        "vadd.u32   d24, d24, d25   \n"
        "vadd.u32   d26, d26, d27   \n"
        "vadd.u32   d28, d28, d29   \n"
        "vadd.u32   d30, d30, d31   \n"
        "vzip.u32   q12, q14    \n"
        "vzip.u32   q13, q15    \n"
        "vzip.u32   q12, q13    \n"
        "vadd.u32   q0, q0, q12     \n"
        
        "veor.u8  q12, q5, q8           \n"
        "veor.u8  q13, q5, q9           \n"
        "veor.u8  q14, q5, q10           \n"
        "veor.u8  q15, q5, q11           \n"
        "vcnt.u8  q12, q12          \n"
        "vcnt.u8  q13, q13          \n"
        "vcnt.u8  q14, q14          \n"
        "vcnt.u8  q15, q15          \n"
        "vpaddl.u8 q12, q12         \n"
        "vpaddl.u8 q13, q13         \n"
        "vpaddl.u8 q14, q14         \n"
        "vpaddl.u8 q15, q15         \n"
        "vpaddl.u16 q12, q12            \n"
        "vpaddl.u16 q13, q13            \n"
        "vpaddl.u16 q14, q14            \n"
        "vpaddl.u16 q15, q15            \n"
        "vpaddl.u32 q12, q12            \n"
        "vpaddl.u32 q13, q13            \n"
        "vpaddl.u32 q14, q14            \n"
        "vpaddl.u32 q15, q15            \n"
        "vadd.u32   d24, d24, d25   \n"
        "vadd.u32   d26, d26, d27   \n"
        "vadd.u32   d28, d28, d29   \n"
        "vadd.u32   d30, d30, d31   \n"
        "vzip.u32   q12, q14    \n"
        "vzip.u32   q13, q15    \n"
        "vzip.u32   q12, q13    \n"
        "vadd.u32   q1, q1, q12     \n"
        
        "veor.u8  q12, q6, q8           \n"
        "veor.u8  q13, q6, q9           \n"
        "veor.u8  q14, q6, q10           \n"
        "veor.u8  q15, q6, q11           \n"
        "vcnt.u8  q12, q12          \n"
        "vcnt.u8  q13, q13          \n"
        "vcnt.u8  q14, q14          \n"
        "vcnt.u8  q15, q15          \n"
        "vpaddl.u8 q12, q12         \n"
        "vpaddl.u8 q13, q13         \n"
        "vpaddl.u8 q14, q14         \n"
        "vpaddl.u8 q15, q15         \n"
        "vpaddl.u16 q12, q12            \n"
        "vpaddl.u16 q13, q13            \n"
        "vpaddl.u16 q14, q14            \n"
        "vpaddl.u16 q15, q15            \n"
        "vpaddl.u32 q12, q12            \n"
        "vpaddl.u32 q13, q13            \n"
        "vpaddl.u32 q14, q14            \n"
        "vpaddl.u32 q15, q15            \n"
        "vadd.u32   d24, d24, d25   \n"
        "vadd.u32   d26, d26, d27   \n"
        "vadd.u32   d28, d28, d29   \n"
        "vadd.u32   d30, d30, d31   \n"
        "vzip.u32   q12, q14    \n"
        "vzip.u32   q13, q15    \n"
        "vzip.u32   q12, q13    \n"
        "vadd.u32   q2, q2, q12     \n"
        
        "subs %0, %0, #1    \n"

        "veor.u8  q12, q7, q8           \n"
        "veor.u8  q13, q7, q9           \n"
        "veor.u8  q14, q7, q10           \n"
        "veor.u8  q15, q7, q11           \n"
        "vcnt.u8  q12, q12          \n"
        "vcnt.u8  q13, q13          \n"
        "vcnt.u8  q14, q14          \n"
        "vcnt.u8  q15, q15          \n"
        "vpaddl.u8 q12, q12         \n"
        "vpaddl.u8 q13, q13         \n"
        "vpaddl.u8 q14, q14         \n"
        "vpaddl.u8 q15, q15         \n"
        "vpaddl.u16 q12, q12            \n"
        "vpaddl.u16 q13, q13            \n"
        "vpaddl.u16 q14, q14            \n"
        "vpaddl.u16 q15, q15            \n"
        "vpaddl.u32 q12, q12            \n"
        "vpaddl.u32 q13, q13            \n"
        "vpaddl.u32 q14, q14            \n"
        "vpaddl.u32 q15, q15            \n"
        "vadd.u32   d24, d24, d25   \n"
        "vadd.u32   d26, d26, d27   \n"
        "vadd.u32   d28, d28, d29   \n"
        "vadd.u32   d30, d30, d31   \n"
        "vzip.u32   q12, q14    \n"
        "vzip.u32   q13, q15    \n"
        "vzip.u32   q12, q13    \n"
        "vadd.u32   q3, q3, q12     \n"
        "bne 0b     \n"

        "vcvt.u32.f32  q0, q0   \n"
        "vcvt.u32.f32  q1, q1   \n"
        "vcvt.u32.f32  q2, q2   \n"
        "vcvt.u32.f32  q3, q3   \n"
        "vst1.32    q0, [%1]!   \n"
        "vst1.32    q1, [%1]!   \n"
        "vst1.32    q2, [%1]!   \n"
        "vst1.32    q3, [%1]!   \n"
        : "+r"(kc),  // %0
          "+r"(c),   // %1
          "+r"(b),   // %2
          "+r"(a)    // %3
        :
        : "cc", "memory", "r0", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
        );
        /*
        "ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x0], #64    \n"
        "ld1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x0], #64    \n"
        "ld1 {v8.4s, v9.4s, v10.4s, v11.4s}, [x0]    \n"

        "0: \n"
        "ld1 {v18.2d, v19.2d, v20.2d, v21.2d}, [%3], #64    \n"
        "ld1 {v12.2d, v13.2d, v14.2d, v15.2d}, [%2], #64    \n"
        "eor v26.16b, v12.16b, v18.16b         \n"
        "eor v27.16b, v12.16b, v19.16b         \n"
        "eor v28.16b, v12.16b, v20.16b         \n"
        "eor v29.16b, v12.16b, v21.16b         \n"
        "cnt v26.16b, v26.16b                 \n"
        "cnt v27.16b, v27.16b                 \n"
        "cnt v28.16b, v28.16b                 \n"
        "cnt v29.16b, v29.16b                 \n"
        "ld1 {v22.2d, v23.2d, v24.2d, v25.2d}, [%3], #64    \n"
        "addv b26, v26.16b                \n"
        "addv b27, v27.16b                \n"
        "addv b28, v28.16b                \n"
        "addv b29, v29.16b                \n"
        "ins v26.s[1], v27.s[0]             \n"
        "ins v26.s[2], v28.s[0]             \n"
        "ins v26.s[3], v29.s[0]             \n"

        "ld1 {v16.2d, v17.2d}, [%2], #32    \n"
        "eor v27.16b, v12.16b, v22.16b         \n"
        "eor v28.16b, v12.16b, v23.16b         \n"
        "eor v29.16b, v12.16b, v24.16b         \n"
        "eor v30.16b, v12.16b, v25.16b         \n"
        "cnt v27.16b, v27.16b                 \n"
        "cnt v28.16b, v28.16b                 \n"
        "cnt v29.16b, v29.16b                 \n"
        "cnt v30.16b, v30.16b                 \n"
        "addv b27, v27.16b                \n"
        "addv b28, v28.16b                \n"
        "addv b29, v29.16b                \n"
        "addv b30, v30.16b                \n"
        "ins v27.s[1], v28.s[0]             \n"
        "ins v27.s[2], v29.s[0]             \n"
        "ins v27.s[3], v30.s[0]             \n"

        "add v0.4s, v0.4s, v26.4s           \n"  // delay
        "add v1.4s, v1.4s, v27.4s           \n"

        // The No.1 col of C is finished

        "eor v26.16b, v13.16b, v18.16b         \n"
        "eor v27.16b, v13.16b, v19.16b         \n"
        "eor v28.16b, v13.16b, v20.16b         \n"
        "eor v29.16b, v13.16b, v21.16b         \n"
        "cnt v26.16b, v26.16b                 \n"
        "cnt v27.16b, v27.16b                 \n"
        "cnt v28.16b, v28.16b                 \n"
        "cnt v29.16b, v29.16b                 \n"
        "addv b26, v26.16b                \n"
        "addv b27, v27.16b                \n"
        "addv b28, v28.16b                \n"
        "addv b29, v29.16b                \n"
        "ins v26.s[1], v27.s[0]             \n"
        "ins v26.s[2], v28.s[0]             \n"
        "ins v26.s[3], v29.s[0]             \n"

        "eor v27.16b, v13.16b, v22.16b         \n"
        "eor v28.16b, v13.16b, v23.16b         \n"
        "eor v29.16b, v13.16b, v24.16b         \n"
        "eor v30.16b, v13.16b, v25.16b         \n"
        "cnt v27.16b, v27.16b                 \n"
        "cnt v28.16b, v28.16b                 \n"
        "cnt v29.16b, v29.16b                 \n"
        "cnt v30.16b, v30.16b                 \n"
        "addv b27, v27.16b                \n"
        "addv b28, v28.16b                \n"
        "addv b29, v29.16b                \n"
        "addv b30, v30.16b                \n"
        "ins v27.s[1], v28.s[0]             \n"
        "ins v27.s[2], v29.s[0]             \n"
        "ins v27.s[3], v30.s[0]             \n"

        "add v2.4s, v2.4s, v26.4s           \n"  // delay
        "add v3.4s, v3.4s, v27.4s           \n"

        // The No.2 col of C is finished

        "eor v26.16b, v14.16b, v18.16b         \n"
        "eor v27.16b, v14.16b, v19.16b         \n"
        "eor v28.16b, v14.16b, v20.16b         \n"
        "eor v29.16b, v14.16b, v21.16b         \n"
        "cnt v26.16b, v26.16b                 \n"
        "cnt v27.16b, v27.16b                 \n"
        "cnt v28.16b, v28.16b                 \n"
        "cnt v29.16b, v29.16b                 \n"
        "addv b26, v26.16b                \n"
        "addv b27, v27.16b                \n"
        "addv b28, v28.16b                \n"
        "addv b29, v29.16b                \n"
        "ins v26.s[1], v27.s[0]             \n"
        "ins v26.s[2], v28.s[0]             \n"
        "ins v26.s[3], v29.s[0]             \n"

        "eor v27.16b, v14.16b, v22.16b         \n"
        "eor v28.16b, v14.16b, v23.16b         \n"
        "eor v29.16b, v14.16b, v24.16b         \n"
        "eor v30.16b, v14.16b, v25.16b         \n"
        "cnt v27.16b, v27.16b                 \n"
        "cnt v28.16b, v28.16b                 \n"
        "cnt v29.16b, v29.16b                 \n"
        "cnt v30.16b, v30.16b                 \n"
        "addv b27, v27.16b                \n"
        "addv b28, v28.16b                \n"
        "addv b29, v29.16b                \n"
        "addv b30, v30.16b                \n"
        "ins v27.s[1], v28.s[0]             \n"
        "ins v27.s[2], v29.s[0]             \n"
        "ins v27.s[3], v30.s[0]             \n"

        "add v4.4s, v4.4s, v26.4s           \n"  // delay
        "add v5.4s, v5.4s, v27.4s           \n"

        // The No.3 col of C is finished

        "prfm   pldl1keep, [%3, #128]     \n"
        "eor v26.16b, v15.16b, v18.16b         \n"
        "eor v27.16b, v15.16b, v19.16b         \n"
        "eor v28.16b, v15.16b, v20.16b         \n"
        "eor v29.16b, v15.16b, v21.16b         \n"
        "cnt v26.16b, v26.16b                 \n"
        "cnt v27.16b, v27.16b                 \n"
        "cnt v28.16b, v28.16b                 \n"
        "cnt v29.16b, v29.16b                 \n"
        "addv b26, v26.16b                \n"
        "addv b27, v27.16b                \n"
        "addv b28, v28.16b                \n"
        "addv b29, v29.16b                \n"
        "ins v26.s[1], v27.s[0]             \n"
        "ins v26.s[2], v28.s[0]             \n"
        "ins v26.s[3], v29.s[0]             \n"

        "prfm   pldl1keep, [%2, #128]     \n"
        "eor v27.16b, v15.16b, v22.16b         \n"
        "eor v28.16b, v15.16b, v23.16b         \n"
        "eor v29.16b, v15.16b, v24.16b         \n"
        "eor v30.16b, v15.16b, v25.16b         \n"
        "cnt v27.16b, v27.16b                 \n"
        "cnt v28.16b, v28.16b                 \n"
        "cnt v29.16b, v29.16b                 \n"
        "cnt v30.16b, v30.16b                 \n"
        "addv b27, v27.16b                \n"
        "addv b28, v28.16b                \n"
        "addv b29, v29.16b                \n"
        "addv b30, v30.16b                \n"
        "ins v27.s[1], v28.s[0]             \n"
        "ins v27.s[2], v29.s[0]             \n"
        "ins v27.s[3], v30.s[0]             \n"

        "add v6.4s, v6.4s, v26.4s           \n"  // delay
        "add v7.4s, v7.4s, v27.4s           \n"

        // The No.4 col of C is finished

        "eor v26.16b, v16.16b, v18.16b         \n"
        "eor v27.16b, v16.16b, v19.16b         \n"
        "eor v28.16b, v16.16b, v20.16b         \n"
        "eor v29.16b, v16.16b, v21.16b         \n"
        "cnt v26.16b, v26.16b                 \n"
        "cnt v27.16b, v27.16b                 \n"
        "cnt v28.16b, v28.16b                 \n"
        "cnt v29.16b, v29.16b                 \n"
        "addv b26, v26.16b                \n"
        "addv b27, v27.16b                \n"
        "addv b28, v28.16b                \n"
        "addv b29, v29.16b                \n"
        "ins v26.s[1], v27.s[0]             \n"
        "ins v26.s[2], v28.s[0]             \n"
        "ins v26.s[3], v29.s[0]             \n"

        "eor v27.16b, v16.16b, v22.16b         \n"
        "eor v28.16b, v16.16b, v23.16b         \n"
        "eor v29.16b, v16.16b, v24.16b         \n"
        "eor v30.16b, v16.16b, v25.16b         \n"
        "cnt v27.16b, v27.16b                 \n"
        "cnt v28.16b, v28.16b                 \n"
        "cnt v29.16b, v29.16b                 \n"
        "cnt v30.16b, v30.16b                 \n"
        "addv b27, v27.16b                \n"
        "addv b28, v28.16b                \n"
        "addv b29, v29.16b                \n"
        "addv b30, v30.16b                \n"
        "ins v27.s[1], v28.s[0]             \n"
        "ins v27.s[2], v29.s[0]             \n"
        "ins v27.s[3], v30.s[0]             \n"

        "add v8.4s, v8.4s, v26.4s           \n"  // delay
        "add v9.4s, v9.4s, v27.4s           \n"

        // The No.5 col of C is finished

        "eor v26.16b, v17.16b, v18.16b         \n"
        "eor v27.16b, v17.16b, v19.16b         \n"
        "eor v28.16b, v17.16b, v20.16b         \n"
        "eor v29.16b, v17.16b, v21.16b         \n"
        "cnt v26.16b, v26.16b                 \n"
        "cnt v27.16b, v27.16b                 \n"
        "cnt v28.16b, v28.16b                 \n"
        "cnt v29.16b, v29.16b                 \n"
        "addv b26, v26.16b                \n"
        "addv b27, v27.16b                \n"
        "addv b28, v28.16b                \n"
        "addv b29, v29.16b                \n"
        "ins v26.s[1], v27.s[0]             \n"
        "ins v26.s[2], v28.s[0]             \n"
        "ins v26.s[3], v29.s[0]             \n"

        "subs %0, %0, #1    \n"

        "eor v27.16b, v17.16b, v22.16b         \n"
        "eor v28.16b, v17.16b, v23.16b         \n"
        "eor v29.16b, v17.16b, v24.16b         \n"
        "eor v30.16b, v17.16b, v25.16b         \n"
        "cnt v27.16b, v27.16b                 \n"
        "cnt v28.16b, v28.16b                 \n"
        "cnt v29.16b, v29.16b                 \n"
        "cnt v30.16b, v30.16b                 \n"
        "addv b27, v27.16b                \n"
        "addv b28, v28.16b                \n"
        "addv b29, v29.16b                \n"
        "addv b30, v30.16b                \n"
        "ins v27.s[1], v28.s[0]             \n"
        "ins v27.s[2], v29.s[0]             \n"
        "ins v27.s[3], v30.s[0]             \n"

        "add v10.4s, v10.4s, v26.4s           \n"  // delay
        "add v11.4s, v11.4s, v27.4s           \n"

        // The No.6 col of C is finished

        "bne 0b     \n"

        "ucvtf  v0.4s, v0.4s    \n"
        "ucvtf  v1.4s, v1.4s    \n"
        "ucvtf  v2.4s, v2.4s    \n"
        "ucvtf  v3.4s, v3.4s    \n"
        "ucvtf  v4.4s, v4.4s    \n"
        "ucvtf  v5.4s, v5.4s    \n"
        "st1 {v0.4s}, [%1], #16     \n"
        "ucvtf  v6.4s, v6.4s    \n"
        "st1 {v1.4s}, [%1], #16     \n"
        "ucvtf  v7.4s, v7.4s    \n"
        "st1 {v2.4s}, [%1], #16     \n"
        "ucvtf  v8.4s, v8.4s    \n"
        "st1 {v3.4s}, [%1], #16     \n"
        "ucvtf  v9.4s, v9.4s    \n"
        "st1 {v4.4s}, [%1], #16     \n"
        "ucvtf  v10.4s, v10.4s    \n"
        "st1 {v5.4s}, [%1], #16     \n"
        "ucvtf  v11.4s, v11.4s    \n"
        "st1 {v6.4s}, [%1], #16     \n"
        "st1 {v7.4s}, [%1], #16     \n"
        "st1 {v8.4s}, [%1], #16     \n"
        "st1 {v9.4s}, [%1], #16     \n"
        "st1 {v10.4s}, [%1], #16     \n"
        "st1 {v11.4s}, [%1], #16     \n"
        : "+r"(kc),  // %0
          "+r"(c),   // %1
          "+r"(b),   // %2
          "+r"(a)    // %3
        :
        : "cc", "memory", "x0", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
          "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17",
          "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27",
          "v28", "v29", "v30");
    */

#endif  // __aarch64__
}
#endif  // __ARM_NEON

inline void bgemm_naive(const int m, const int n, const int k,
                        const uint64_t *a, const int lda, const uint64_t *b,
                        const int ldb, float *c, const int ldc) {
    FORZ(i, m) {
        FORZ(j, n) {
            FORZ(h, k) {
                C(i, j) += static_cast<float>(bitcount((A(i, h) ^ B(h, j))));
            }
        }
    }
}

#undef R
#undef P
#undef A
#undef B
#undef C
#undef min
#endif /* BNN_BGEMM_H */

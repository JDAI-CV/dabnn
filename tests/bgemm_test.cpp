// Copyright 2019 JD.com Inc. JD AI

#include <common/baseline.h>
#include <dabnn/bgemm.h>
#include <dabnn/im2col.h>
#include <dabnn/mat.h>

#include <gtest/gtest.h>

#include <common/helper.h>

TEST(bgemm, bgemm) {
    const int m = 159;
    const int n = 253;
    const int k = 68;

    uint64_t a[m * k];
    uint64_t b[k * n];
    fill_rand_uint64(a, m * k);
    fill_rand_uint64(b, k * n);
    float c[m * n] = {};
    float c_navie[m * n] = {};
    bgemm(m, n, k, a, m, b, k, c, m);
    bgemm_naive(m, n, k, a, m, b, k, c_navie, m);

    ASSERT_EQ(std::memcmp(c, c_navie, sizeof(c)), 0);
}

/**
 * Test the edge cause of the input/output size is very small.
 */
TEST(bgemm, bconv2) {
    const size_t AHEIGHT = 3;
    const size_t AWIDTH = 3;
    const size_t CHANNEL = 128;

    const size_t BHEIGHT = 3;
    const size_t BWIDTH = 3;
    const size_t NUM_OUTPUT = 128;

    const size_t CHEIGHT = 1;
    const size_t CWIDTH = 1;

    const size_t ALEN = AHEIGHT * AWIDTH * CHANNEL / sizeof(uint64_t);
    const size_t BLEN =
        NUM_OUTPUT * BHEIGHT * BWIDTH * CHANNEL / sizeof(uint64_t);

    uint64_t a_data[ALEN];
    uint64_t b_data[BLEN];
    fill_rand_uint64(a_data, ALEN);
    fill_rand_uint64(b_data, BLEN);

    const bnn::Mat a(AHEIGHT, AWIDTH, CHANNEL, a_data, bnn::DataType::Bit);
    const bnn::Mat b(NUM_OUTPUT, BHEIGHT, BWIDTH, CHANNEL, b_data,
                     bnn::DataType::Bit, false);
    bnn::Mat expected(CHEIGHT, CWIDTH, NUM_OUTPUT, bnn::DataType::Float);
    expected.fill<uint32_t>(0);
    bnn::baseline_bconv(a, b, 3, 3, 0, 0, 2, 2, 1, 1, NUM_OUTPUT, expected);

    const int m = NUM_OUTPUT;
    const int n = CHEIGHT * CWIDTH;
    const int k = BHEIGHT * BWIDTH * CHANNEL / 64;
    uint64_t b_data_[BLEN];
    FORZ(i, k) {
        FORZ(j, m) { b_data_[i * m + j] = b_data[j * k + i]; }
    }
    memcpy(b_data, b_data_, BLEN * 8);

    float c_data[m * n] = {};
    bnn::Mat a_col(CHEIGHT * CWIDTH * BHEIGHT * BWIDTH * CHANNEL,
                   bnn::DataType::Bit);
    bnn::im2col(a, BHEIGHT, BWIDTH, 0, 0, 1, 1, 1, 1, a_col);

    bnn::Mat c(CHEIGHT, CWIDTH, NUM_OUTPUT, c_data, bnn::DataType::Float);
    bgemm(m, n, k, static_cast<uint64_t *>(b.data), m,
          static_cast<uint64_t *>(a_col.data), k, static_cast<float *>(c.data),
          m);

    ASSERT_EQ(c, expected);
}

TEST(bgemm, bconv) {
    const size_t AHEIGHT = 64;
    const size_t AWIDTH = 64;
    const size_t CHANNEL = 128;

    const size_t BHEIGHT = 5;
    const size_t BWIDTH = 5;
    const size_t NUM_OUTPUT = 128;

    const size_t CHEIGHT = AHEIGHT / 2;
    const size_t CWIDTH = AWIDTH / 2;

    const size_t ALEN = AHEIGHT * AWIDTH * CHANNEL / sizeof(uint64_t);
    const size_t BLEN =
        NUM_OUTPUT * BHEIGHT * BWIDTH * CHANNEL / sizeof(uint64_t);

    uint64_t a_data[ALEN];
    uint64_t b_data[BLEN];
    fill_rand_uint64(a_data, ALEN);
    fill_rand_uint64(b_data, BLEN);

    const bnn::Mat a(AHEIGHT, AWIDTH, CHANNEL, a_data, bnn::DataType::Bit);
    const bnn::Mat b(NUM_OUTPUT, BHEIGHT, BWIDTH, CHANNEL, b_data,
                     bnn::DataType::Bit, false);
    bnn::Mat expected(CHEIGHT, CWIDTH, NUM_OUTPUT, bnn::DataType::Float);
    expected.fill<uint32_t>(0);
    bnn::baseline_bconv(a, b, 5, 5, 2, 2, 2, 2, 1, 1, NUM_OUTPUT, expected);

    const int m = NUM_OUTPUT;
    const int n = CHEIGHT * CWIDTH;
    const int k = BHEIGHT * BWIDTH * CHANNEL / 64;
    uint64_t b_data_[BLEN];
    FORZ(i, k) {
        FORZ(j, m) { b_data_[i * m + j] = b_data[j * k + i]; }
    }
    memcpy(b_data, b_data_, BLEN * 8);

    float c_data[m * n] = {};
    bnn::Mat a_col(CHEIGHT * CWIDTH * BHEIGHT * BWIDTH * CHANNEL,
                   bnn::DataType::Bit);
    bnn::im2col(a, BHEIGHT, BWIDTH, 2, 2, 2, 2, 1, 1, a_col);

    bnn::Mat c(CHEIGHT, CWIDTH, NUM_OUTPUT, c_data, bnn::DataType::Float);
    bgemm(m, n, k, static_cast<uint64_t *>(b.data), m,
          static_cast<uint64_t *>(a_col.data), k, static_cast<float *>(c.data),
          m);

    ASSERT_EQ(c, expected);
}

TEST(bgemm, bgemm_128) {
    const int m = 128;
    const int n = 28 * 28;
    const int k = 18;

    uint64_t a[m * k];
    uint64_t b[k * n];
    fill_rand_uint64(a, m * k);
    fill_rand_uint64(b, k * n);
    float c[m * n] = {};
    float c_navie[m * n] = {};
    bgemm(m, n, k, a, m, b, k, c, m);
    bgemm_naive(m, n, k, a, m, b, k, c_navie, m);

    ASSERT_EQ(std::memcmp(c, c_navie, sizeof(c)), 0);
}

TEST(bgemm, bgemm_256) {
    const int m = 256;
    const int n = 14 * 14;
    const int k = 36;

    uint64_t a[m * k];
    uint64_t b[k * n];
    fill_rand_uint64(a, m * k);
    fill_rand_uint64(b, k * n);
    float c[m * n] = {};
    float c_navie[m * n] = {};
    bgemm(m, n, k, a, m, b, k, c, m);
    bgemm_naive(m, n, k, a, m, b, k, c_navie, m);

    ASSERT_EQ(std::memcmp(c, c_navie, sizeof(c)), 0);
}

TEST(bgemm, bgemm_512) {
    const int m = 512;
    const int n = 7 * 7;
    const int k = 72;

    uint64_t a[m * k];
    uint64_t b[k * n];
    fill_rand_uint64(a, m * k);
    fill_rand_uint64(b, k * n);
    float c[m * n] = {};
    float c_navie[m * n] = {};
    bgemm(m, n, k, a, m, b, k, c, m);
    bgemm_naive(m, n, k, a, m, b, k, c_navie, m);

    ASSERT_EQ(std::memcmp(c, c_navie, sizeof(c)), 0);
}

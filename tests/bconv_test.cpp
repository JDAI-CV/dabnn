// Copyright 2019 JD.com Inc. JD AI

#include <dabnn/bconv.h>

#include <bitset>
#include <chrono>

#include <common/baseline.h>
#include <common/helper.h>
#include <dabnn/bitpack.h>
#include <dabnn/pad.h>
#include <gtest/gtest.h>

/*
// TODO: reuse the code
TEST(bconv_test, bconv_test_1x1_unroll_64) {
    const size_t AHEIGHT = 64;
    const size_t AWIDTH = 64;
    const size_t CHANNEL = 64;

    const size_t BHEIGHT = 1;
    const size_t BWIDTH = 1;
    const size_t NUM_OUTPUT = 256;

    const size_t CHEIGHT = AHEIGHT - BHEIGHT + 1;
    const size_t CWIDTH = AWIDTH - BWIDTH + 1;

    const size_t ALEN = AHEIGHT * AWIDTH * CHANNEL / sizeof(uint64_t);
    const size_t BLEN = NUM_OUTPUT * BHEIGHT * BWIDTH * CHANNEL /
sizeof(uint64_t);

    uint64_t a_data[ALEN];
    uint64_t b_data[BLEN];
    fill_rand_uint64(a_data, ALEN);
    fill_rand_uint64(b_data, BLEN);

    const bnn::Mat a(AHEIGHT, AWIDTH, CHANNEL, a_data, bnn::DataType::Bit);
    const bnn::Mat b(NUM_OUTPUT, BHEIGHT, BWIDTH, CHANNEL, b_data,
bnn::DataType::Bit, 0, false);

    bnn::Mat c(CHEIGHT, CWIDTH, NUM_OUTPUT, bnn::DataType::Float);
    c.fill<uint32_t>(0);
    bnn::bconv_1x1_unroll4_64(a, b, c);

    bnn::Mat expected(CHEIGHT, CWIDTH, NUM_OUTPUT, bnn::DataType::Float);
    expected.fill<uint32_t>(0);
    bnn::baseline_bconv2(a, b, 1, 1, 0, 0, 1, 1, 1, 1, NUM_OUTPUT, expected);
    // baseline_bconv(a, b, expected);

    ASSERT_EQ(c, expected);
}

TEST(bconv_test, bconv_test_1x1_unroll_128) {
    const size_t AHEIGHT = 64;
    const size_t AWIDTH = 64;
    const size_t CHANNEL = 128;

    const size_t BHEIGHT = 1;
    const size_t BWIDTH = 1;
    const size_t NUM_OUTPUT = 256;

    const size_t CHEIGHT = AHEIGHT - BHEIGHT + 1;
    const size_t CWIDTH = AWIDTH - BWIDTH + 1;

    const size_t ALEN = AHEIGHT * AWIDTH * CHANNEL / sizeof(uint64_t);
    const size_t BLEN = NUM_OUTPUT * BHEIGHT * BWIDTH * CHANNEL /
sizeof(uint64_t);

    uint64_t a_data[ALEN];
    uint64_t b_data[BLEN];
    fill_rand_uint64(a_data, ALEN);
    fill_rand_uint64(b_data, BLEN);

    const bnn::Mat a(AHEIGHT, AWIDTH, CHANNEL, a_data, bnn::DataType::Bit);
    const bnn::Mat b(NUM_OUTPUT, BHEIGHT, BWIDTH, CHANNEL, b_data,
bnn::DataType::Bit, 0, false);

    bnn::Mat c(CHEIGHT, CWIDTH, NUM_OUTPUT, bnn::DataType::Float);
    c.fill<uint32_t>(0);
    bnn::bconv_1x1_unroll4_128(a, b, c);

    bnn::Mat expected(CHEIGHT, CWIDTH, NUM_OUTPUT, bnn::DataType::Float);
    expected.fill<uint32_t>(0);
    bnn::baseline_bconv2(a, b, 1, 1, 0, 0, 1, 1, 1, 1, NUM_OUTPUT, expected);

    ASSERT_EQ(c, expected);
}
*/

/*
TEST(bconv_test, bconv_test_1x1_unroll_256) {
    const size_t AHEIGHT = 64;
    const size_t AWIDTH = 64;
    const size_t CHANNEL = 256;

    const size_t BHEIGHT = 1;
    const size_t BWIDTH = 1;
    const size_t NUM_OUTPUT = 256;

    const size_t CHEIGHT = AHEIGHT - BHEIGHT + 1;
    const size_t CWIDTH = AWIDTH - BWIDTH + 1;

    const size_t ALEN = AHEIGHT * AWIDTH * CHANNEL / sizeof(uint64_t);
    const size_t BLEN = NUM_OUTPUT * BHEIGHT * BWIDTH * CHANNEL /
sizeof(uint64_t);

    uint64_t a_data[ALEN];
    uint64_t b_data[BLEN];
    fill_rand_uint64(a_data, ALEN);
    fill_rand_uint64(b_data, BLEN);

    const bnn::Mat a(AHEIGHT, AWIDTH, CHANNEL, a_data, bnn::DataType::Bit);
    const bnn::Mat b(NUM_OUTPUT, BHEIGHT, BWIDTH, CHANNEL, b_data,
bnn::DataType::Bit, 0, false);

    bnn::Mat c(CHEIGHT, CWIDTH, NUM_OUTPUT, bnn::DataType::Float);
    c.fill<uint32_t>(0);
    bnn::bconv_1x1_unroll4_256(a, b, c);

    bnn::Mat expected(CHEIGHT, CWIDTH, NUM_OUTPUT, bnn::DataType::Float);
    expected.fill<uint32_t>(0);
    baseline_bconv(a, b, expected);

    ASSERT_EQ(c, expected);
}

TEST(bconv_test, bconv_test_1x1_unroll_512) {
    const size_t AHEIGHT = 64;
    const size_t AWIDTH = 64;
    const size_t CHANNEL = 512;

    const size_t BHEIGHT = 1;
    const size_t BWIDTH = 1;
    const size_t NUM_OUTPUT = 512;

    const size_t CHEIGHT = AHEIGHT - BHEIGHT + 1;
    const size_t CWIDTH = AWIDTH - BWIDTH + 1;

    const size_t ALEN = AHEIGHT * AWIDTH * CHANNEL / sizeof(uint64_t);
    const size_t BLEN = NUM_OUTPUT * BHEIGHT * BWIDTH * CHANNEL /
sizeof(uint64_t);

    uint64_t a_data[ALEN];
    uint64_t b_data[BLEN];
    fill_rand_uint64(a_data, ALEN);
    fill_rand_uint64(b_data, BLEN);

    const bnn::Mat a(AHEIGHT, AWIDTH, CHANNEL, a_data, bnn::DataType::Bit);
    const bnn::Mat b(NUM_OUTPUT, BHEIGHT, BWIDTH, CHANNEL, b_data,
bnn::DataType::Bit, 0, false);

    bnn::Mat c(CHEIGHT, CWIDTH, NUM_OUTPUT, bnn::DataType::Float);
    c.fill<uint32_t>(0);
    bnn::bconv_1x1_unroll4_512(a, b, c);

    bnn::Mat expected(CHEIGHT, CWIDTH, NUM_OUTPUT, bnn::DataType::Float);
    expected.fill<uint32_t>(0);
    baseline_bconv(a, b, expected);

    ASSERT_EQ(c, expected);
}
*/

TEST(bconv_test, bconv_test_3x3_64) {
    const size_t AHEIGHT = 64;
    const size_t AWIDTH = 64;
    const size_t CHANNEL = 64;

    const size_t BHEIGHT = 3;
    const size_t BWIDTH = 3;
    const size_t NUM_OUTPUT = 64;

    const size_t CHEIGHT = AHEIGHT;
    const size_t CWIDTH = AWIDTH;

    const size_t ALEN = AHEIGHT * AWIDTH * CHANNEL / sizeof(uint64_t);
    const size_t BLEN =
        NUM_OUTPUT * BHEIGHT * BWIDTH * CHANNEL / sizeof(uint64_t);

    uint64_t a_data[ALEN];
    uint64_t b_data[BLEN];
    fill_rand_uint64(a_data, ALEN);
    fill_rand_uint64(b_data, BLEN);

    const bnn::Mat a(AHEIGHT, AWIDTH, CHANNEL, a_data, bnn::DataType::Bit);
    bnn::Mat padded(AHEIGHT + 2, AWIDTH + 2, CHANNEL, bnn::DataType::Bit);
    pad(a, 1, 1, padded);
    const bnn::Mat b(NUM_OUTPUT, BHEIGHT, BWIDTH, CHANNEL, b_data,
                     bnn::DataType::Bit, false);

    bnn::Mat c(CHEIGHT, CWIDTH, NUM_OUTPUT, bnn::DataType::Float);
    c.fill<float>(0);
    bnn::bconv_3x3_64(padded, b, c);

    bnn::Mat expected(CHEIGHT, CWIDTH, NUM_OUTPUT, bnn::DataType::Float);
    expected.fill<float>(0);
    bnn::baseline_bconv(a, b, 3, 3, 1, 1, 1, 1, 1, 1, NUM_OUTPUT, expected);
    // baseline_bconv(a, b, expected);

    ASSERT_EQ(c, expected);
}

TEST(bconv_test, bconv_test_3x3_64_s2) {
    const size_t AHEIGHT = 64;
    const size_t AWIDTH = 64;
    const size_t CHANNEL = 64;

    const size_t BHEIGHT = 3;
    const size_t BWIDTH = 3;
    const size_t NUM_OUTPUT = 64;

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
    bnn::Mat padded(AHEIGHT + 2, AWIDTH + 2, CHANNEL, bnn::DataType::Bit);
    pad(a, 1, 1, padded);
    const bnn::Mat b(NUM_OUTPUT, BHEIGHT, BWIDTH, CHANNEL, b_data,
                     bnn::DataType::Bit, false);

    bnn::Mat c(CHEIGHT, CWIDTH, NUM_OUTPUT, bnn::DataType::Float);
    c.fill<float>(0);
    bnn::bconv_3x3_64(padded, b, c, 2);

    bnn::Mat expected(CHEIGHT, CWIDTH, NUM_OUTPUT, bnn::DataType::Float);
    expected.fill<float>(0);
    bnn::baseline_bconv(a, b, 3, 3, 1, 1, 2, 2, 1, 1, NUM_OUTPUT, expected);

    ASSERT_EQ(c, expected);
}

TEST(bconv_test, bconv_test_3x3_128) {
    const size_t AHEIGHT = 64;
    const size_t AWIDTH = 64;
    const size_t CHANNEL = 128;

    const size_t BHEIGHT = 3;
    const size_t BWIDTH = 3;
    const size_t NUM_OUTPUT = 128;

    const size_t CHEIGHT = AHEIGHT;
    const size_t CWIDTH = AWIDTH;

    const size_t ALEN = AHEIGHT * AWIDTH * CHANNEL / sizeof(uint64_t);
    const size_t BLEN =
        NUM_OUTPUT * BHEIGHT * BWIDTH * CHANNEL / sizeof(uint64_t);

    uint64_t a_data[ALEN];
    uint64_t b_data[BLEN];
    fill_rand_uint64(a_data, ALEN);
    fill_rand_uint64(b_data, BLEN);

    const bnn::Mat a(AHEIGHT, AWIDTH, CHANNEL, a_data, bnn::DataType::Bit);
    bnn::Mat padded(AHEIGHT + 2, AWIDTH + 2, CHANNEL, bnn::DataType::Bit);
    pad(a, 1, 1, padded);
    const bnn::Mat b(NUM_OUTPUT, BHEIGHT, BWIDTH, CHANNEL, b_data,
                     bnn::DataType::Bit, false);

    bnn::Mat c(CHEIGHT, CWIDTH, NUM_OUTPUT, bnn::DataType::Float);
    c.fill<uint32_t>(0);
    bnn::bconv_3x3(padded, b, c, 1);

    bnn::Mat expected(CHEIGHT, CWIDTH, NUM_OUTPUT, bnn::DataType::Float);
    expected.fill<uint32_t>(0);
    bnn::baseline_bconv(a, b, 3, 3, 1, 1, 1, 1, 1, 1, NUM_OUTPUT, expected);
    // baseline_bconv(a, b, expected);

    ASSERT_EQ(c, expected);
}

TEST(bconv_test, bconv_test_3x3_128_s2) {
    const size_t AHEIGHT = 64;
    const size_t AWIDTH = 64;
    const size_t CHANNEL = 128;

    const size_t BHEIGHT = 3;
    const size_t BWIDTH = 3;
    const size_t NUM_OUTPUT = 256;

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
    bnn::Mat padded(AHEIGHT + 2, AWIDTH + 2, CHANNEL, bnn::DataType::Bit);
    pad(a, 1, 1, padded);
    const bnn::Mat b(NUM_OUTPUT, BHEIGHT, BWIDTH, CHANNEL, b_data,
                     bnn::DataType::Bit, false);

    bnn::Mat c(CHEIGHT, CWIDTH, NUM_OUTPUT, bnn::DataType::Float);
    c.fill<uint32_t>(0);
    bnn::bconv_3x3(padded, b, c, 2);

    bnn::Mat expected(CHEIGHT, CWIDTH, NUM_OUTPUT, bnn::DataType::Float);
    expected.fill<uint32_t>(0);
    bnn::baseline_bconv(a, b, 3, 3, 1, 1, 2, 2, 1, 1, NUM_OUTPUT, expected);
    // baseline_bconv(a, b, expected);

    ASSERT_EQ(c, expected);
}

TEST(bconv_test, bconv_test_3x3_256) {
    const size_t AHEIGHT = 64;
    const size_t AWIDTH = 64;
    const size_t CHANNEL = 256;

    const size_t BHEIGHT = 3;
    const size_t BWIDTH = 3;
    const size_t NUM_OUTPUT = 256;

    const size_t CHEIGHT = AHEIGHT;
    const size_t CWIDTH = AWIDTH;

    const size_t ALEN = AHEIGHT * AWIDTH * CHANNEL / sizeof(uint64_t);
    const size_t BLEN =
        NUM_OUTPUT * BHEIGHT * BWIDTH * CHANNEL / sizeof(uint64_t);

    uint64_t a_data[ALEN];
    uint64_t b_data[BLEN];
    fill_rand_uint64(a_data, ALEN);
    fill_rand_uint64(b_data, BLEN);

    const bnn::Mat a(AHEIGHT, AWIDTH, CHANNEL, a_data, bnn::DataType::Bit);
    bnn::Mat padded(AHEIGHT + 2, AWIDTH + 2, CHANNEL, bnn::DataType::Bit);
    pad(a, 1, 1, padded);
    const bnn::Mat b(NUM_OUTPUT, BHEIGHT, BWIDTH, CHANNEL, b_data,
                     bnn::DataType::Bit, false);

    bnn::Mat c(CHEIGHT, CWIDTH, NUM_OUTPUT, bnn::DataType::Float);
    c.fill<uint32_t>(0);
    bnn::bconv_3x3(padded, b, c);

    bnn::Mat expected(CHEIGHT, CWIDTH, NUM_OUTPUT, bnn::DataType::Float);
    expected.fill<uint32_t>(0);
    bnn::baseline_bconv(a, b, 3, 3, 1, 1, 1, 1, 1, 1, NUM_OUTPUT, expected);

    ASSERT_EQ(c, expected);
}

TEST(bconv_test, bconv_test_3x3_512) {
    const size_t AHEIGHT = 64;
    const size_t AWIDTH = 64;
    const size_t CHANNEL = 512;

    const size_t BHEIGHT = 3;
    const size_t BWIDTH = 3;
    const size_t NUM_OUTPUT = 256;

    const size_t CHEIGHT = AHEIGHT;
    const size_t CWIDTH = AWIDTH;

    const size_t ALEN = AHEIGHT * AWIDTH * CHANNEL / sizeof(uint64_t);
    const size_t BLEN =
        NUM_OUTPUT * BHEIGHT * BWIDTH * CHANNEL / sizeof(uint64_t);

    uint64_t a_data[ALEN];
    uint64_t b_data[BLEN];
    fill_rand_uint64(a_data, ALEN);
    fill_rand_uint64(b_data, BLEN);

    const bnn::Mat a(AHEIGHT, AWIDTH, CHANNEL, a_data, bnn::DataType::Bit);
    bnn::Mat padded(AHEIGHT + 2, AWIDTH + 2, CHANNEL, bnn::DataType::Bit);
    pad(a, 1, 1, padded);
    const bnn::Mat b(NUM_OUTPUT, BHEIGHT, BWIDTH, CHANNEL, b_data,
                     bnn::DataType::Bit, false);

    bnn::Mat c(CHEIGHT, CWIDTH, NUM_OUTPUT, bnn::DataType::Float);
    c.fill<uint32_t>(0);
    bnn::bconv_3x3(padded, b, c);

    bnn::Mat expected(CHEIGHT, CWIDTH, NUM_OUTPUT, bnn::DataType::Float);
    expected.fill<uint32_t>(0);
    bnn::baseline_bconv(a, b, 3, 3, 1, 1, 1, 1, 1, 1, NUM_OUTPUT, expected);

    ASSERT_EQ(c, expected);
}

TEST(bconv_test, bconv_test_3x3_512_s2) {
    const size_t AHEIGHT = 64;
    const size_t AWIDTH = 64;
    const size_t CHANNEL = 512;

    const size_t BHEIGHT = 3;
    const size_t BWIDTH = 3;
    const size_t NUM_OUTPUT = 1024;

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
    bnn::Mat padded(AHEIGHT + 2, AWIDTH + 2, CHANNEL, bnn::DataType::Bit);
    pad(a, 1, 1, padded);
    const bnn::Mat b(NUM_OUTPUT, BHEIGHT, BWIDTH, CHANNEL, b_data,
                     bnn::DataType::Bit, false);

    bnn::Mat c(CHEIGHT, CWIDTH, NUM_OUTPUT, bnn::DataType::Float);
    c.fill<uint32_t>(0);
    bnn::bconv_3x3(padded, b, c, 2);

    bnn::Mat expected(CHEIGHT, CWIDTH, NUM_OUTPUT, bnn::DataType::Float);
    expected.fill<uint32_t>(0);
    bnn::baseline_bconv(a, b, 3, 3, 1, 1, 2, 2, 1, 1, NUM_OUTPUT, expected);

    ASSERT_EQ(c, expected);
}

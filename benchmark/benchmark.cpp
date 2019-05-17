// Copyright 2019 JD.com Inc. JD AI

#include <chrono>
#include <memory>

#include <benchmark/benchmark.h>
#include <common/baseline.h>
#include <common/helper.h>
#include <dabnn/bconv.h>
#include <dabnn/bgemm.h>
#include <dabnn/bitpack.h>
#include <dabnn/layers/MaxPool.h>
#include <dabnn/mat.h>
#include <dabnn/net.h>

static void BM_pack_mat_64_small(benchmark::State &state) {
    const bnn::Mat a(1, 32, 32, 128, bnn::DataType::Float, 0);
    bnn::Mat b(1, 32, 32, 128, bnn::DataType::Bit, 0);
    for (auto _ : state) {
        pack_mat_64(a, b);
    }
}

static void BM_pack_mat_128_small(benchmark::State &state) {
    const bnn::Mat a(1, 32, 32, 128, bnn::DataType::Float, 0);
    bnn::Mat b(1, 32, 32, 128, bnn::DataType::Bit, 0);
    for (auto _ : state) {
        pack_mat_128(a, b);
    }
}

static void BM_pack_mat_64(benchmark::State &state) {
    const bnn::Mat a(1, 64, 64, 128, bnn::DataType::Float);
    bnn::Mat b(1, 64, 64, 128, bnn::DataType::Bit);
    for (auto _ : state) {
        pack_mat_64(a, b);
    }
}

static void BM_pack_mat_128(benchmark::State &state) {
    const bnn::Mat a(1, 64, 64, 128, bnn::DataType::Float);
    bnn::Mat b(1, 64, 64, 128, bnn::DataType::Bit);
    for (auto _ : state) {
        pack_mat_128(a, b);
    }
}

#define SETUP_BCONV_FLOAT(size_a, size_b, num_output)                         \
    const size_t AHEIGHT = size_a;                                            \
    const size_t AWIDTH = size_a;                                             \
    const size_t CHANNEL = num_output;                                        \
                                                                              \
    const size_t BHEIGHT = size_b;                                            \
    const size_t BWIDTH = size_b;                                             \
    const size_t NUM_OUTPUT = num_output;                                     \
                                                                              \
    const size_t CHEIGHT = AHEIGHT - BHEIGHT + 1;                             \
    const size_t CWIDTH = AWIDTH - BWIDTH + 1;                                \
                                                                              \
    const size_t ALEN = AHEIGHT * AWIDTH * CHANNEL;                           \
    const size_t BLEN = NUM_OUTPUT * BHEIGHT * BWIDTH * CHANNEL / 64;         \
                                                                              \
    float a_data[ALEN];                                                       \
    uint64_t b_data[BLEN];                                                    \
    FORZ(i, ALEN) { a_data[i] = 3 * i; }                                      \
    FORZ(i, BLEN) { b_data[i] = 2 * i; }                                      \
                                                                              \
    const bnn::Mat a(AHEIGHT, AWIDTH, CHANNEL, a_data, bnn::DataType::Float); \
    const bnn::Mat b(NUM_OUTPUT, BHEIGHT, BWIDTH, CHANNEL, b_data,            \
                     bnn::DataType::Bit);                                     \
                                                                              \
    bnn::Mat a_binary(AHEIGHT, AWIDTH, CHANNEL, bnn::DataType::Bit);          \
                                                                              \
    bnn::Mat c(CHEIGHT, CWIDTH, NUM_OUTPUT, bnn::DataType::Float);

static void BM_bconv_float_3x3_128(benchmark::State &state) {
    SETUP_BCONV_FLOAT(30, 3, 128);
    for (auto _ : state) {
        pack_mat_128(a, a_binary);
        bnn::bconv_3x3(a_binary, b, c);
    }
}

static void BM_bconv_float_1x1_128(benchmark::State &state) {
    SETUP_BCONV_FLOAT(28, 1, 128);
    for (auto _ : state) {
        pack_mat_128(a, a_binary);
        bnn::bconv_1x1_128(a_binary, b, c);
    }
}

#undef SETUP_BCONV_FLOAT

#define SETUP_BCONV(size_a, size_b, num_output, stride)                  \
    const size_t AHEIGHT = size_a;                                       \
    const size_t AWIDTH = size_a;                                        \
    const size_t CHANNEL = num_output / 64;                              \
                                                                         \
    const size_t BHEIGHT = size_b;                                       \
    const size_t BWIDTH = size_b;                                        \
    const size_t NUM_OUTPUT = num_output;                                \
                                                                         \
    const size_t CHEIGHT = (AHEIGHT - BHEIGHT + 1) / stride;             \
    const size_t CWIDTH = (AWIDTH - BWIDTH + 1) / stride;                \
                                                                         \
    const size_t ALEN = AHEIGHT * AWIDTH * CHANNEL;                      \
    const size_t BLEN = NUM_OUTPUT * BHEIGHT * BWIDTH * CHANNEL;         \
                                                                         \
    uint64_t a_data[ALEN];                                               \
    uint64_t b_data[BLEN];                                               \
    FORZ(i, ALEN) { a_data[i] = 3 * i; }                                 \
    FORZ(i, BLEN) { b_data[i] = 2 * i; }                                 \
                                                                         \
    const bnn::Mat a(1, AHEIGHT, AWIDTH, CHANNEL * sizeof(uint64_t) * 8, \
                     a_data, bnn::DataType::Bit);                        \
    const bnn::Mat b(NUM_OUTPUT, BHEIGHT, BWIDTH,                        \
                     CHANNEL * sizeof(uint64_t) * 8, b_data,             \
                     bnn::DataType::Bit, false);                         \
                                                                         \
    bnn::Mat c(1, CHEIGHT, CWIDTH, NUM_OUTPUT, bnn::DataType::Float);

static void BM_bnn_bconv_3x3_naive_128(benchmark::State &state) {
    SETUP_BCONV(30, 3, 128, 1);
    for (auto _ : state) {
        bnn::baseline_bconv(a, b, BHEIGHT, BWIDTH, 0, 0, 1, 1, 1, 1, NUM_OUTPUT,
                            c);
    }
}

static void BM_bnn_bconv_1x1_naive_128(benchmark::State &state) {
    SETUP_BCONV(28, 1, 128, 1);
    for (auto _ : state) {
        bnn::baseline_bconv(a, b, BHEIGHT, BWIDTH, 0, 0, 1, 1, 1, 1, NUM_OUTPUT,
                            c);
    }
}

static void BM_bnn_bconv_1x1_64(benchmark::State &state) {
    SETUP_BCONV(56, 1, 64, 1);
    for (auto _ : state) {
        bnn::bconv_1x1_64(a, b, c);
    }
}

static void BM_bnn_bconv_1x1_128(benchmark::State &state) {
    SETUP_BCONV(28, 1, 128, 1);
    for (auto _ : state) {
        bnn::bconv_1x1_128(a, b, c);
    }
}

static void BM_bnn_bconv_1x1_256(benchmark::State &state) {
    SETUP_BCONV(14, 1, 256, 1);
    for (auto _ : state) {
        bnn::bconv_1x1_256(a, b, c);
    }
}

static void BM_bnn_bconv_1x1_512(benchmark::State &state) {
    SETUP_BCONV(7, 1, 512, 1);
    for (auto _ : state) {
        bnn::bconv_1x1_512(a, b, c);
    }
}

static void BM_bnn_bconv_3x3_64(benchmark::State &state) {
    SETUP_BCONV(58, 3, 64, 1);
    for (auto _ : state) {
        bnn::bconv_3x3(a, b, c);
    }
}

static void BM_bnn_bconv_3x3_128(benchmark::State &state) {
    SETUP_BCONV(30, 3, 128, 1);
    for (auto _ : state) {
        bnn::bconv_3x3(a, b, c);
    }
}

static void BM_bnn_bconv_3x3_256(benchmark::State &state) {
    SETUP_BCONV(16, 3, 256, 1);
    for (auto _ : state) {
        bnn::bconv_3x3(a, b, c);
    }
}

static void BM_bnn_bconv_3x3_256_s2(benchmark::State &state) {
    SETUP_BCONV(16, 3, 256, 2);
    for (auto _ : state) {
        bnn::bconv_3x3(a, b, c, 2);
    }
}

static void BM_bnn_bconv_3x3_512(benchmark::State &state) {
    SETUP_BCONV(9, 3, 512, 1);
    for (auto _ : state) {
        bnn::bconv_3x3(a, b, c);
    }
}

static void BM_bnn_bconv_3x3_1024(benchmark::State &state) {
    SETUP_BCONV(9, 3, 1024, 1);
    for (auto _ : state) {
        bnn::bconv_3x3(a, b, c);
    }
}

static void BM_bnn_bconv(benchmark::State &state) {
    SETUP_BCONV(30, 3, 128, 1);
    for (auto _ : state) {
        bnn::baseline_bconv(a, b, BHEIGHT, BWIDTH, 0, 0, 1, 1, 1, 1, NUM_OUTPUT,
                            c);
    }
}

#undef SETUP_BCONV

#if 0
static void BM_maxpool3x3(benchmark::State &state) {
    const size_t AHEIGHT = 32;
    const size_t AWIDTH = 32;
    const size_t CHANNEL = 128;

    const size_t CHEIGHT = 30;
    const size_t CWIDTH = 30;

    const size_t ALEN = AHEIGHT * AWIDTH * CHANNEL;

    uint64_t a_data[ALEN];
    FORZ(i, ALEN) { a_data[i] = 3 * i; }

    const auto a = std::make_shared<bnn::Mat>(AHEIGHT, AWIDTH, CHANNEL, a_data, bnn::DataType::Float);

    auto c = std::make_shared<bnn::Mat>(CHEIGHT, CWIDTH, CHANNEL, bnn::DataType::Float);

    const auto m = bnn::MaxPool(a, a, c, 3, 3, 0, 0, 1, 1);
    for (auto _ : state) {
        m.forward();
    }
}
#endif

#define SETUP_BGEMM     \
    uint64_t a[102400]; \
    uint64_t b[102400]; \
    float c[602400];

static void BM_bgemm_64(benchmark::State &state) {
    SETUP_BGEMM;
    for (auto _ : state) {
        bgemm(64, 56 * 56, 9, a, 64, b, 9, c, 64);
    }
}

static void BM_bgemm_128(benchmark::State &state) {
    SETUP_BGEMM;
    for (auto _ : state) {
        bgemm(128, 28 * 28, 18, a, 128, b, 18, c, 128);
    }
}

static void BM_bgemm_naive_128(benchmark::State &state) {
    SETUP_BGEMM;
    for (auto _ : state) {
        bgemm_naive(128, 28 * 28, 18, a, 128, b, 18, c, 128);
    }
}

static void BM_bgemm_256(benchmark::State &state) {
    SETUP_BGEMM;
    for (auto _ : state) {
        bgemm(256, 14 * 14, 36, a, 256, b, 36, c, 256);
    }
}

static void BM_bgemm_256_s2(benchmark::State &state) {
    SETUP_BGEMM;
    for (auto _ : state) {
        bgemm(256, 7 * 7, 36, a, 256, b, 36, c, 256);
    }
}

static void BM_bgemm_512(benchmark::State &state) {
    SETUP_BGEMM;
    for (auto _ : state) {
        bgemm(512, 7 * 7, 72, a, 512, b, 72, c, 512);
    }
}

static void BM_bgemm_5x5_256(benchmark::State &state) {
    SETUP_BGEMM;
    for (auto _ : state) {
        bgemm(256, 14 * 14, 100, a, 256, b, 100, c, 256);
    }
}

static void BM_bgemm_naive_256(benchmark::State &state) {
    SETUP_BGEMM;
    for (auto _ : state) {
        bgemm_naive(256, 14 * 14, 36, a, 256, b, 36, c, 256);
    }
}

static void BM_bireal18_cifar(benchmark::State &state) {
    float input[3 * 32 * 32];

    auto net = bnn::Net::create();
    net->read("/data/local/tmp/model_cifar.dab");
    for (auto _ : state) {
        net->run(input);
    }
}

static void BM_bireal18_imagenet(benchmark::State &state) {
    float input[3 * 224 * 224];

    auto net = bnn::Net::create();
    net->read("/data/local/tmp/model_imagenet.dab");
    for (auto _ : state) {
        net->run(input);
    }
}

static void BM_bireal18_imagenet_stem(benchmark::State &state) {
    float input[3 * 224 * 224];

    auto net = bnn::Net::create();
    net->read("/data/local/tmp/model_imagenet_stem.dab");
    for (auto _ : state) {
        net->run(input);
    }
}

static void BM_bireal18_cifar_wo_fconv(benchmark::State &state) {
    float input[3 * 32 * 32];

    auto net = bnn::Net::create();
    net->run_fconv = false;
    net->strict = false;
    net->read("/data/local/tmp/model_cifar.dab");
    for (auto _ : state) {
        net->run(input);
    }
}

static void BM_bireal18_imagenet_wo_fconv(benchmark::State &state) {
    float input[3 * 224 * 224];

    auto net = bnn::Net::create();
    net->run_fconv = false;
    net->strict = false;
    net->read("/data/local/tmp/model_imagenet.dab");
    for (auto _ : state) {
        net->run(input);
    }
}

BENCHMARK_MAIN();

// BENCHMARK(BM_pack_mat_64);
// BENCHMARK(BM_pack_mat_128);
// BENCHMARK(BM_bnn_bconv_1x1_64);
// BENCHMARK(BM_bnn_bconv_1x1_128);
// BENCHMARK(BM_bnn_bconv_1x1_256);
// BENCHMARK(BM_bnn_bconv_1x1_512);
// BENCHMARK(BM_bgemm_128);
// BENCHMARK(BM_bgemm_256);
// BENCHMARK(BM_bgemm_256_s2);
BENCHMARK(BM_bgemm_5x5_256);
// BENCHMARK(BM_bgemm_512);
BENCHMARK(BM_bnn_bconv_3x3_64);
BENCHMARK(BM_bnn_bconv_3x3_128);
BENCHMARK(BM_bnn_bconv_3x3_256);
BENCHMARK(BM_bnn_bconv_3x3_256_s2);
BENCHMARK(BM_bnn_bconv_3x3_512);
// BENCHMARK(BM_bnn_bconv_3x3_1024);
// BENCHMARK(BM_bireal18_cifar_wo_fconv);
// BENCHMARK(BM_bireal18_imagenet_wo_fconv);
// BENCHMARK(BM_bireal18_cifar);
BENCHMARK(BM_bireal18_imagenet);
BENCHMARK(BM_bireal18_imagenet_stem);
// BENCHMARK(BM_bnn_bconv_3x3_naive_128);
// BENCHMARK(BM_bconv_float_1x1_128);
// BENCHMARK(BM_bconv_float_3x3_128);

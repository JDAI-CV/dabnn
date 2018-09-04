// Copyright 2019 JD.com Inc. JD AI

#include <chrono>

#include <common/baseline.h>
#include <common/helper.h>
#include <dabnn/bconv.h>
#include <dabnn/bconv2.h>
#include <dabnn/bitpack.h>
#include <dabnn/mat.h>

int main(int argc, char **argv) {
    if (argc != 2) {
        LOG(INFO) << "argc must be 2, but it is " << argc << " now";
        exit(-1);
    }
    using Clock = std::chrono::system_clock;

    if (std::string(argv[1]) == "0") {
        static uint64_t weight[99999];
        static uint64_t input[9999999];
        const auto t1 = Clock::now();
        FORZ(n, 1000) {
            FORZ(i, 30752) { new_bconv3x3_128(9 / 3, weight, input); }
        }
        const auto t2 = Clock::now();
        PNT(std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1)
                .count());

    } else {
        const size_t AHEIGHT = 64;
        const size_t AWIDTH = 64;
        const size_t CHANNEL = 2;  // 128;

        const size_t BHEIGHT = 3;
        const size_t BWIDTH = 3;
        const size_t NUM_OUTPUT = 128;

        const size_t CHEIGHT = AHEIGHT - BHEIGHT + 1;
        const size_t CWIDTH = AWIDTH - BWIDTH + 1;

        const size_t ALEN = AHEIGHT * AWIDTH * CHANNEL;
        const size_t BLEN = NUM_OUTPUT * BHEIGHT * BWIDTH * CHANNEL;

        uint64_t a_data[ALEN];
        uint64_t b_data[BLEN];
        FORZ(i, ALEN) { a_data[i] = 3 * i; }
        FORZ(i, BLEN) { b_data[i] = 2 * i; }

        const bnn::Mat a(1, AHEIGHT, AWIDTH, CHANNEL * sizeof(uint64_t) * 8,
                         a_data, bnn::DataType::Bit, 0);
        const bnn::Mat b(NUM_OUTPUT, BHEIGHT, BWIDTH,
                         CHANNEL * sizeof(uint64_t) * 8, b_data,
                         bnn::DataType::Bit, 0, false);

        bnn::Mat c(1, CHEIGHT, CWIDTH, NUM_OUTPUT, bnn::DataType::Float);

        const auto t1 = Clock::now();
        FORZ(n, 1000) { bnn::bconv_3x3_128(a, b, c); }
        const auto t2 = Clock::now();
        PNT(std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1)
                .count());
    }
}

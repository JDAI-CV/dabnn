// Copyright 2019 JD.com Inc. JD AI

#include <sys/types.h>
#include <unistd.h>
#include <algorithm>
#include <chrono>

#include <common/argh.h>
#include <common/flatbuffers_helper.h>
#include <dabnn/net.h>

int main(int argc, char **argv) {
    argh::parser cmdl(argc, argv);
    google::InitGoogleLogging(argv[0]);
    cmdl("v", 1) >> FLAGS_v;
    FLAGS_alsologtostderr = true;
    // FLAGS_logbuflevel = -1;

    float *input = new float[3 * 224 * 224];
    FORZ(i, 3 * 224 * 224) { input[i] = 1; }

    auto net1 = bnn::Net::create();
    net1->optimize = true;
    net1->run_fconv = true;
    net1->strict = true;
    net1->read(argv[1]);
    FORZ(i, 0) { net1->run(input); }
    const int N = 1;
    using Clock = std::chrono::steady_clock;
    const auto t1 = Clock::now();
    FORZ(i, N) {
        LOG(INFO) << "------";
        net1->run(input);
    }
    const auto t2 = Clock::now();
    css blob_name = argv[2];
    LOG(INFO) << "Fetching blob: " << blob_name;
    const auto &blob1 = net1->get_blob(blob_name);
    LOG(INFO) << blob1->total();
    if (blob1->data_type == bnn::DataType::Float) {
        blob1->dump("/data/local/tmp/mat.txt");
    }
    FORZ(i, std::min(static_cast<int>(blob1->total()), 10)) {
        if (blob1->data_type == bnn::DataType::Float) {
            LOG(INFO) << static_cast<float *>(blob1->data)[i];
        } else {
            LOG(INFO) << binrep(static_cast<uint64_t *>(blob1->data) + i, 64, false);
        }
    }
    LOG(INFO) << "Time: "
              << 1.f *
                     std::chrono::duration_cast<std::chrono::nanoseconds>(t2 -
                                                                          t1)
                         .count() /
                     N / 1000000000;
#ifdef BNN_BENCHMARK
    net1->print_time();
#endif

    /*
    bnn::Net net2;
    net2.model_ = model;
    net2.prepare();
    LOG(INFO) << "-----";
    net2.optimize = false;

    net2.run(input);
    const auto &blob2 = net2.get_blob(blob_name);
    LOG(INFO) << blob2->total();
    FORZ(i, std::min(static_cast<int>(blob2->total()), 10)) {
        LOG(INFO) << static_cast<float *>(blob2->data)[i];
    }

    const bool eq = (*blob1 == *blob2);
    BNN_ASSERT(eq, "");
    */
}

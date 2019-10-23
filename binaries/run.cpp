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

    float input[3 * 224 * 224];
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

    for (int i = 2; i < cmdl.size(); i++) {
        css blob_name = argv[i];
        LOG(INFO) << "Fetching blob: " << blob_name;
        const auto &blob1 = net1->get_blob(blob_name);
        LOG(INFO) << static_cast<float *>(blob1->data)[0];
        if (blob1->data_type == bnn::DataType::Float) {
            blob1->dump("/data/local/tmp/mat_" + blob_name + ".txt");
        }
        FORZ(j, std::min(static_cast<int>(blob1->total()), 10)) {
            if (blob1->data_type == bnn::DataType::Float) {
                LOG(INFO) << blob_name << ": " << static_cast<float *>(blob1->data)[j];
            } else {
                LOG(INFO) << blob_name << ": " << binrep(static_cast<uint64_t *>(blob1->data) + j, 64, true);
            }
        }
    }
#ifdef BNN_BENCHMARK
    net1->print_time();
#endif
}

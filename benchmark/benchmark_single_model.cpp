#include <string>

#include <benchmark/benchmark.h>

#include <dabnn/net.h>

static void BM_single_model(benchmark::State &state, std::string model_path) {
    float input[999999];

    auto net = bnn::Net::create();
    net->optimize = true;
    net->run_fconv = true;
    net->strict = true;
    net->read(model_path);

    for (auto _ : state) {
        net->run(input);
    }
}

int main(int argc, char **argv) {
    benchmark::RegisterBenchmark("single_model", BM_single_model, argv[1]);
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
}

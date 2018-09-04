// Copyright 2019 JD.com Inc. JD AI

#include "layer.h"

#include <chrono>

#include "net.h"

namespace bnn {
Layer::~Layer() {}

Layer::MatCP Layer::mat(const std::string &name) const {
    return net_.lock()->get_blob(name);
}

void Layer::forward() {
#ifdef BNN_BENCHMARK
    const auto start = std::chrono::system_clock::now();
#endif
    forward_impl();
#ifdef BNN_BENCHMARK
    const auto end = std::chrono::system_clock::now();
    const auto elapsed_time_double =
        1. *
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count() /
        1000;
    const auto elapsed_time = std::to_string(elapsed_time_double) + "ms";

    net_.lock()->layer_time_[type_] += elapsed_time_double;
#endif
}

std::string Layer::to_str() const { return ""; }

}  // namespace bnn

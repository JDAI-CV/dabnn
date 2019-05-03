# Copyright 2019 JD.com Inc. JD AI

add_executable(dabnn_benchmark benchmark.cpp)
target_link_libraries(dabnn_benchmark
    dabnn
    benchmark_main)

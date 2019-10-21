# Copyright 2019 JD.com Inc. JD AI

add_executable(dabnn_benchmark benchmark.cpp)
target_link_libraries(dabnn_benchmark
    dabnn
    benchmark_main)

add_executable(benchmark_single_model benchmark_single_model.cpp)
target_link_libraries(benchmark_single_model
    dabnn
    benchmark_main)

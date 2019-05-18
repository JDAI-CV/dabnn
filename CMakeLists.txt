# Copyright 2019 JD.com Inc. JD AI

cmake_minimum_required(VERSION 3.5)
project(dabnn CXX)

message(STATUS "CMake Version: ${CMAKE_VERSION}")

if (NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
    message(STATUS "No build type, set to Release")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(BNN_BUILD_JNI "Build jni" OFF)
option(BNN_BUILD_TEST "Build test" ON)
option(BNN_BUILD_BENCHMARK "Build benchmark" ON)
option(BNN_NET_BENCHMARK "define BNN_BENCHMARK" OFF)
option(BNN_CHECK_CONSISTENCY "Make some layers non inplace to check results" OFF)

if (${CMAKE_SYSTEM_NAME} STREQUAL "Android")
    set(BNN_BUILD_ANDROID ON)
else()
    set(BNN_BUILD_ANDROID OFF)
endif()

include(cmake/common.cmake)
include(cmake/system.cmake)

include(cmake/glog.cmake)
configure_glog()

add_compile_options("-DEIGEN_MPL2_ONLY")
if (${BNN_NET_BENCHMARK})
    add_compile_options("-DBNN_BENCHMARK")
endif()

if (${BNN_BUILD_ANDROID})
    set(CMAKE_CXX_STANDARD 17)

    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

    if (${BNN_BUILD_TEST})
        include(cmake/gtest.cmake)
        configure_gtest()
        add_subdirectory(third_party/googletest)
    endif()

    add_subdirectory(dabnn)

    add_subdirectory(binaries)
    if (BNN_BUILD_TEST)
        add_subdirectory(tests)
    endif()
    if (BNN_BUILD_BENCHMARK)
        include(cmake/benchmark.cmake)
        configure_benchmark()
        add_subdirectory(benchmark)
    endif()

else()
    set(CMAKE_CXX_STANDARD 11)

    include(cmake/onnx.cmake)
    configure_onnx()
    add_subdirectory(tools)
endif()

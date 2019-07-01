# Copyright 2019 JD.com Inc. JD AI

cmake_minimum_required(VERSION 3.9)
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
option(BNN_SYSTEM_PROTOBUF "Use system protobuf to build onnx2bnn" ON)
option(BNN_BUILD_PYTHON "Build onnx2bnn python interface" OFF)
option(BNN_USE_MSVC_STATIC_RUNTIME "Link onnx2bnn to msvc static runtime" ON)

message(STATUS "Target architecture: ${CMAKE_SYSTEM_PROCESSOR}")
if (NOT DEFINED BNN_BUILD_MAIN_LIB)
    if (CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64" OR 
            CMAKE_SYSTEM_PROCESSOR STREQUAL "armv7-a")
        set(BNN_BUILD_MAIN_LIB ON)
    else()
        set(BNN_BUILD_MAIN_LIB OFF)
    endif()
endif()
message(STATUS "BNN_BUILD_MAIN_LIB: ${BNN_BUILD_MAIN_LIB}")

include(cmake/utils.cmake)
bnn_add_msvc_runtime_flag()

include(cmake/common.cmake)
include(cmake/system.cmake)

include(cmake/glog.cmake)
configure_glog()

add_compile_options("-DEIGEN_MPL2_ONLY")
if (${BNN_NET_BENCHMARK})
    add_compile_options("-DBNN_BENCHMARK")
endif()

if (BNN_BUILD_MAIN_LIB)
    set(CMAKE_CXX_STANDARD 17)

    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

    if (${BNN_BUILD_TEST})
        include(cmake/gtest.cmake)
        configure_gtest()
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

    if (${BNN_BUILD_PYTHON})
        set(CMAKE_POSITION_INDEPENDENT_CODE ON)
        add_subdirectory(third_party/pybind11)
    endif()

    include(cmake/protobuf.cmake)
    if (${BNN_SYSTEM_PROTOBUF})
        find_package(Protobuf)
    endif()
    if (NOT Protobuf_FOUND)
        message(STATUS "BNN_SYSTEM_PROTOBUF is ON but protobuf is not found, using the packed protobuf..")
        set(BNN_SYSTEM_PROTOBUF OFF CACHE BOOL "Use system protobuf when building onnx2daq" FORCE)
    endif()

    include(cmake/onnx.cmake)
    configure_onnx()

    add_subdirectory(tools)
endif()

# Copyright 2019 JD.com Inc. JD AI

add_library(dabnn
    allocator.cpp
    allocator.h
    mat.h
    bconv.h
    bitpack.h
    net.cpp
    im2col.h
    fconv.h
    layers/FloatConv.cpp
    layers/FloatConv.h
    layers/BinConv.cpp
    layers/BinConv.h
    layers/MaxPool.cpp
    layers/MaxPool.h
    layers/Affine.cpp
    layers/Affine.h
    layers/AvePool.cpp
    layers/AvePool.h
    layers/Add.cpp
    layers/Add.h
    layers/Concat.cpp
    layers/Concat.h
    layers/Relu.cpp
    layers/Relu.h
    layers/Binarize.cpp
    layers/Binarize.h
    layers/Shuffle.cpp
    layers/Shuffle.h
    layers/Split.cpp
    layers/Split.h
    layers/PRelu.cpp
    layers/PRelu.h
    layer.h
    layer.cpp
    ${PROJECT_SOURCE_DIR}/common/Shaper.cpp
    ${PROJECT_SOURCE_DIR}/common/Shaper.h
    )
target_include_directories(dabnn
    PUBLIC
    ${CMAKE_CURRENT_BINARY_DIR}
    ${PROJECT_SOURCE_DIR}
    )
target_include_directories(dabnn
    SYSTEM
    PUBLIC
    ${PROJECT_SOURCE_DIR}/third_party/eigen
    )
target_link_libraries(dabnn
    glog::glog
    flatbuffers
    )

if (${BNN_BUILD_JNI})
    add_subdirectory(jni)
endif()

if (${BNN_CHECK_CONSISTENCY})
    target_compile_definitions(
        dabnn
        PRIVATE
        BNN_CHECK_CONSISTENCY
        )
endif()

treat_warnings_as_errors(dabnn)

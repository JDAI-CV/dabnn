# Copyright 2019 JD.com Inc. JD AI

find_package(Protobuf)

add_executable(onnx2bnn
    ${CMAKE_CURRENT_SOURCE_DIR}/onnx2bnn.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/OnnxConverter.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/OnnxConverter.h
    ${CMAKE_CURRENT_SOURCE_DIR}/NodeAttrHelper.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/NodeAttrHelper.h
    ${PROJECT_SOURCE_DIR}/common/StrKeyMap.h
    ${PROJECT_SOURCE_DIR}/common/Shaper.h
    ${PROJECT_SOURCE_DIR}/common/Shaper.cpp
    )

target_link_libraries(onnx2bnn
    glog::glog
    protobuf::libprotobuf
    onnx
    )

target_include_directories(onnx2bnn
    PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
    )

treat_warnings_as_errors(onnx2bnn)

# Copyright 2019 JD.com Inc. JD AI

include(${PROJECT_SOURCE_DIR}/cmake/utils.cmake)

add_library(onnx2bnn
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
    onnx
    protobuf::libprotobuf
    )
target_include_directories(onnx2bnn
    PUBLIC
    ${PROJECT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
    )

add_executable(onnx2bnn-bin
    ${CMAKE_CURRENT_SOURCE_DIR}/onnx2bnn.cpp
    )
target_link_libraries(onnx2bnn-bin
    onnx2bnn)
set_target_properties(onnx2bnn-bin PROPERTIES OUTPUT_NAME "onnx2bnn")
if (${BNN_SYSTEM_PROTOBUF})
    treat_warnings_as_errors(onnx2bnn)
    treat_warnings_as_errors(onnx2bnn-bin)
endif()

if (${BNN_BUILD_PYTHON})
    pybind11_add_module(_onnx2bnn
        ${CMAKE_CURRENT_SOURCE_DIR}/pywrapper.cpp)

    target_link_libraries(_onnx2bnn
        PUBLIC
        protobuf::libprotobuf
        glog::glog
        onnx
        onnx2bnn
        pybind11::pybind11
        )
endif()

# Copyright 2019 JD.com Inc. JD AI

find_package(Protobuf REQUIRED)
if (${CMAKE_VERSION} VERSION_LESS "3.6.0")
    set(Protobuf_LIBRARIES ${PROTOBUF_LIBRARIES})
    set(Protobuf_INCLUDE_DIRS ${PROTOBUF_INCLUDE_DIRS})
endif()

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
    onnx
    )

target_include_directories(onnx2bnn
    PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
    )

if(${CMAKE_VERSION} VERSION_GREATER "3.10.0") 
    target_link_libraries(onnx2bnn
        protobuf::libprotobuf)
else()
    target_link_libraries(onnx2bnn
        ${Protobuf_LIBRARIES})
    target_include_directories(onnx2bnn
        PRIVATE
        ${Protobuf_INCLUDE_DIRS})
endif()

treat_warnings_as_errors(onnx2bnn)

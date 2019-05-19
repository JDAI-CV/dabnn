# Copyright 2019 JD.com Inc. JD AI

macro(configure_onnx)
    if (NOT ${BNN_SYSTEM_PROTOBUF})
        include(${PROJECT_SOURCE_DIR}/cmake/protobuf.cmake)
        configure_protobuf()
    endif()

    message(STATUS "Configuring onnx...")
    set(BNN_ONNX_NAMESPACE onnx_bnn)
    if (MSVC)
        set(ONNX_CMAKELISTS ${PROJECT_SOURCE_DIR}/third_party/onnx/CMakeLists.txt)
        file(READ ${ONNX_CMAKELISTS} content)
        # Treating warnings as errors breaks ci, we have no other way to opt-out
        string(
            REPLACE
            "/WX"
            ""
            content
            "${content}"
            )
        file(WRITE ${ONNX_CMAKELISTS} "${content}")
    endif()
    set(ONNX_USE_MSVC_STATIC_RUNTIME ${BNN_USE_MSVC_STATIC_RUNTIME})
    set(ONNX_NAMESPACE ${BNN_ONNX_NAMESPACE} CACHE STRING "onnx namespace")
    add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/onnx)
    # Since https://github.com/onnx/onnx/pull/1318 is merged, we don't need to set it manually
    # target_compile_definitions(onnx
    # PUBLIC
    # -DONNX_NAMESPACE=${BNN_ONNX_NAMESPACE})
endmacro()

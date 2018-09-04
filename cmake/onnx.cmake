# Copyright 2019 JD.com Inc. JD AI

macro(configure_onnx)
    message(STATUS "Configuring onnx...")
    set(BNN_ONNX_NAMESPACE onnx_bnn)
    
    set(ONNX_NAMESPACE ${BNN_ONNX_NAMESPACE} CACHE STRING "onnx namespace")
    add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/onnx)
    # Since https://github.com/onnx/onnx/pull/1318 is merged, we don't need to set it manually
    # target_compile_definitions(onnx
    # PUBLIC
    # -DONNX_NAMESPACE=${BNN_ONNX_NAMESPACE})
endmacro()

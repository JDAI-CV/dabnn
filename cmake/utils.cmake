# Copyright 2019 JD.com Inc. JD AI

# Add MSVC RunTime Flag
function(add_msvc_runtime_flag lib)
    if (MSVC)
        if(${DNN_USE_MSVC_STATIC_RUNTIME})
            if(${CMAKE_BUILD_TYPE} MATCHES "Debug")
                target_compile_options(${lib} PRIVATE /MTd)
            else()
                target_compile_options(${lib} PRIVATE /MT)
            endif()
        else()
            if(${CMAKE_BUILD_TYPE} MATCHES "Debug")
                target_compile_options(${lib} PRIVATE /MDd)
            else()
                target_compile_options(${lib} PRIVATE /MD)
            endif()
        endif()
    endif()
endfunction()




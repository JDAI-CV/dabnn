# Copyright 2019 JD.com Inc. JD AI

# Add MSVC RunTime Flag
function(bnn_add_msvc_runtime_flag)
    if (MSVC)
        if(${BNN_USE_MSVC_STATIC_RUNTIME})
            if(${CMAKE_BUILD_TYPE} MATCHES "Debug")
                add_compile_options(/MTd)
            else()
                add_compile_options(/MT)
            endif()
        else()
            if(${CMAKE_BUILD_TYPE} MATCHES "Debug")
                add_compile_options(/MDd)
            else()
                add_compile_options(/MD)
            endif()
        endif()
    endif()
endfunction()




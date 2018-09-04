# Copyright 2019 JD.com Inc. JD AI

function(treat_warnings_as_errors target)
    if(MSVC)
        target_compile_options(${target} PRIVATE "/W4" "/WX")
    else()
        target_compile_options(${target} PRIVATE "-Wall" "-Wextra" "-Werror")
    endif()
endfunction()

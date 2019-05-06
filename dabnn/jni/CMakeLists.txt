# Copyright 2019 JD.com Inc. JD AI

add_library(dabnn-jni
    SHARED
    jni_handle.h
    net_for_jni.h
    JavaWrapper.cpp
    net_for_jni.cpp
    )

find_library(
    android-lib
    android 
    )

target_link_libraries(
    dabnn-jni
    dabnn
    ${android-lib}
    )

treat_warnings_as_errors(dabnn-jni)

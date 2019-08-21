function(configure_flatbuffers)
    option(FLATBUFFERS_BUILD_TESTS "Enable the build of tests and samples." OFF)
    option(FLATBUFFERS_BUILD_FLATHASH "Enable the build of flathash" OFF)
    option(FLATBUFFERS_BUILD_FLATC "Enable the build of the flatbuffers compiler"
        OFF)
    option(FLATBUFFERS_BUILD_FLATLIB "Enable the build of the flatbuffers library"
        ON)
    add_subdirectory(third_party/flatbuffers)
endfunction()


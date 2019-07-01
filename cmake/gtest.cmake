# Copyright 2019 JD.com Inc. JD AI

macro(configure_gtest)
    message(STATUS "Configuring gtest...")
    option(BUILD_GMOCK "Builds the googlemock subproject" OFF)
    option(INSTALL_GTEST "Enable installation of googletest. (Projects embedding googletest may want to turn this OFF.)" OFF)
    add_subdirectory(third_party/googletest)
endmacro()

# Copyright 2019 JD.com Inc. JD AI

macro(configure_gtest)
    message(STATUS "Configureing gtest...")
    option(BUILD_GMOCK "Builds the googlemock subproject" OFF)
    option(INSTALL_GTEST "Enable installation of googletest. (Projects embedding googletest may want to turn this OFF.)" OFF)
endmacro()

# Copyright 2019 JD.com Inc. JD AI

add_executable(bconv_test bconv_test.cpp)
target_link_libraries(bconv_test dabnn gtest_main)
add_test(NAME bconv_test COMMAND bconv_test)

add_executable(bitpack_test bitpack_test.cpp)
target_link_libraries(bitpack_test dabnn gtest_main)
add_test(NAME bitpack_test COMMAND bitpack_test)

add_executable(im2col_test im2col_test.cpp)
target_link_libraries(im2col_test dabnn gtest_main)
add_test(NAME im2col_test COMMAND im2col_test)

add_executable(fconv_test fconv_test.cpp)
target_link_libraries(fconv_test dabnn gtest_main)
add_test(NAME fconv_test COMMAND fconv_test)

add_executable(bgemm_test bgemm_test.cpp)
target_link_libraries(bgemm_test dabnn gtest_main)
add_test(NAME bgemm_test COMMAND bgemm_test)

add_executable(net_test net_test.cpp)
target_link_libraries(net_test dabnn gtest_main)
add_test(NAME net_test COMMAND net_test)


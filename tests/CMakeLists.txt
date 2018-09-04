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

add_executable(pool_test pool_test.cpp)
target_link_libraries(pool_test dabnn gtest_main)
add_test(NAME pool_test COMMAND pool_test)

add_executable(gemm_test gemm_test.cpp)
target_link_libraries(gemm_test dabnn gtest_main)
add_test(NAME gemm_test COMMAND gemm_test)

add_executable(mat_test mat_test.cpp)
target_link_libraries(mat_test dabnn gtest_main)
add_test(NAME mat_test COMMAND mat_test)

add_executable(pad_test pad_test.cpp)
target_link_libraries(pad_test dabnn gtest_main)
add_test(NAME pad_test COMMAND pad_test)

add_executable(bgemm_test bgemm_test.cpp)
target_link_libraries(bgemm_test dabnn gtest_main)
add_test(NAME bgemm_test COMMAND bgemm_test)

add_executable(net_test net_test.cpp)
target_link_libraries(net_test dabnn gtest_main)
add_test(NAME net_test COMMAND net_test)


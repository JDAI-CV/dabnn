// Copyright 2019 JD.com Inc. JD AI

#include <dabnn/ave_pool.h>

#include <gtest/gtest.h>

#include <common/helper.h>
#include <common/log_helper.h>
#include <dabnn/mat.h>

namespace bnn {

TEST(pool, pool) {
    const size_t len = 32;
    float data[len];
    for (size_t i = 0; i < len; i++) {
        data[i] = i;
    }
    Mat im(4, 4, 2, data, DataType::Float);
    Mat pooled(2, 2, 2, DataType::Float);
    ave_pool(im, 1, 1, 2, 2, 3, 3, pooled);
    float data_expected[]{5, 6, 8, 9, 17, 18, 20, 21};

    Mat expected(2, 2, 2, data_expected, DataType::Float);
    ASSERT_EQ(pooled.flatten(), expected.flatten());
}

}  // namespace bnn

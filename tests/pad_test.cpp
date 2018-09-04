// Copyright 2019 JD.com Inc. JD AI

#include <dabnn/pad.h>

#include <dabnn/mat.h>

#include <gtest/gtest.h>

namespace bnn {
TEST(pad_test, pad) {
    Mat m(4, 4, 2, DataType::Float);
    m.fill<float>(1);
    Mat sm(6, 6, 2, DataType::Float);
    pad(m, 1, 1, sm);

    PNT(m);
    PNT(sm);
}
}  // namespace bnn

// Copyright 2019 JD.com Inc. JD AI

#include <dabnn/mat.h>

#include <gtest/gtest.h>

namespace bnn {
TEST(mat_test, submat) {
    Mat m(4, 4, 2, DataType::Float);
    PNT(m.data);
    Mat sm = m.subMat(2, 3, 2, 4);
    PNT(sm.data);
    FORZ(i, sm.h) {
        auto piece = sm.piece<float>(0, i);
        PNT(piece);
        FORZ(j, sm.w) {
            FORZ(k, sm.c) {
                auto data = piece + j * sm.c + k;
                PNT(data);
                *data = 1;
            }
        }
    }
    PNT(m);
}
}  // namespace bnn

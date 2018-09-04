// Copyright 2019 JD.com Inc. JD AI

#include <Eigen/Dense>

#include <gtest/gtest.h>

#include <common/helper.h>

TEST(gemm, gemm) {
    using namespace Eigen;
    {
        float A[10] = {1.1f,  2.01f,  3.001f,  4.0001f, 5.1f,
                       6.01f, 7.001f, 8.0001f, 9.f,     10.f};
        float B[4] = {1.f, 2.f, 3.f, 4.f};  //, 1.f, 1.f};

        // cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans,4,2,2,1,A, 4, B,
        // 2,0,C,4);
        Map<MatrixXf> a_eg(A, 5, 2);
        Map<MatrixXf> b_eg(B, 2, 2);
        MatrixXf c_eg = a_eg * b_eg;
        // cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans,5,1,2,1,A, 5, B,
        // 1,0,C,5);

        PNT(c_eg);
    }

    /*
    {
    float A[6] = {1.0,2.0,1.0,-3.0,4.0,-1.0};
    float B[6] = {1.0,8.0,7.0,-5.0,6.0,-3.0};
    float C[9] = {.5,.5,.5,.5,.5,.5,.5,.5,.5};
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans,3,3,2,1,A, 3, B,
    3,2,C,3);

    for(i=0; i<9; i++)
        printf("%f ", C[i]);
    printf("\n");
    }
    */
}

// Copyright 2019 JD.com Inc. JD AI

#include <dabnn/fconv.h>

#include <gtest/gtest.h>

#include <common/baseline.h>
#include <common/helper.h>
#include <common/log_helper.h>
#include <dabnn/mat.h>

namespace bnn {

TEST(fconv, fconv) {
    const size_t len = 32;
    float data[len];
    for (size_t i = 0; i < len; i++) {
        data[i] = i;
    }
    Mat im(4, 4, 2, data, DataType::Float);

    const size_t w_len = 3 * 3 * 2 * 2;
    float weight_data[w_len];
    for (size_t i = 0; i < w_len; i++) {
        weight_data[i] = i;
    }
    Mat weight(w_len, weight_data, DataType::Float);

    const size_t output_len = 32;
    Mat output(output_len, DataType::Float);
    Mat output_expected(4, 4, 2, DataType::Float);
    for (size_t i = 0; i < output_len; i++) {
        output_expected[i] = 0;
    }

    fconv(im, weight, 3, 3, 1, 1, 1, 1, 1, 1, 2, output);

    baseline_fconv(im, weight, 3, 3, 1, 1, 1, 1, 1, 1, 2, output_expected);

    ASSERT_EQ(output, output_expected.flatten());
}

TEST(fconv, fconv_stride) {
    const size_t len = 128;
    float data[len];
    for (size_t i = 0; i < len; i++) {
        data[i] = i;
    }
    Mat im(8, 8, 2, data, DataType::Float);

    const size_t w_len = 5 * 5 * 2 * 2;
    float weight_data[w_len];
    for (size_t i = 0; i < w_len; i++) {
        weight_data[i] = i;
    }
    Mat weight(w_len, weight_data, DataType::Float);

    const size_t output_len = 32;
    Mat output(output_len, DataType::Float);
    Mat output_expected(4, 4, 2, DataType::Float);
    for (size_t i = 0; i < output_len; i++) {
        output_expected[i] = 0;
    }

    fconv(im, weight, 5, 5, 2, 2, 2, 2, 1, 1, 2, output);

    baseline_fconv(im, weight, 5, 5, 2, 2, 2, 2, 1, 1, 2, output_expected);

    ASSERT_EQ(output, output_expected.flatten());
}

TEST(fconv, fconv2) {
    const size_t len = 200;
    float data[len];
    for (size_t i = 0; i < len; i++) {
        data[i] = random_float();
    }
    Mat im(10, 10, 2, data, DataType::Float);

    const size_t w_len = 3 * 3 * 2 * 2;
    float weight_data[w_len];
    for (size_t i = 0; i < w_len; i++) {
        weight_data[i] = random_float();
    }
    Mat weight(w_len, weight_data, DataType::Float);

    // TODO: Auto calculate output shape
    const size_t output_len = 72;
    Mat output(output_len, DataType::Float);
    Mat output_expected(6, 6, 2, DataType::Float);
    for (size_t i = 0; i < output_len; i++) {
        output_expected[i] = 0;
    }

    fconv(im, weight, 3, 3, 2, 2, 2, 2, 1, 1, 2, output);

    baseline_fconv(im, weight, 3, 3, 2, 2, 2, 2, 1, 1, 2, output_expected);

    ASSERT_EQ(output, output_expected.flatten());
}

TEST(fconv, fconv3) {
    const int in_c = 4;
    const int out_c = 4;
    const int h = 4;
    const int w = 4;
    const int k_h = 3;
    const int k_w = 3;
    const size_t len = h * w * in_c;
    float data[len];
    fill_rand_float(data, len);
    Mat im(h, w, in_c, data, DataType::Float);

    const size_t w_len = out_c * k_h * k_w * in_c;
    float weight_data[w_len];
    fill_rand_float(weight_data, w_len);
    // for (size_t i = 0; i < w_len; i++) {
    // weight_data[i] = i + ;
    // }
    Mat weight(w_len, weight_data, DataType::Float);

    // TODO: Auto calculate output shape
    const size_t output_len = h * w * out_c;
    Mat output(output_len, DataType::Float);
    Mat output_expected(h, w, out_c, DataType::Float);
    for (size_t i = 0; i < output_len; i++) {
        output_expected[i] = 0;
    }

    fconv(im, weight, k_h, k_w, 1, 1, 1, 1, 1, 1, out_c, output);

    baseline_fconv(im, weight, k_h, k_w, 1, 1, 1, 1, 1, 1, out_c,
                   output_expected);

    /*
    FORZ(i, output_expected.total()) {
        if (std::abs(output[i] - output_expected[i]) > 1e-5) {
            LOG(INFO) << "No. " << i << " not equal, output[i] = " << output[i]
    << ", expected[i] = " << output_expected[i];
        }
    }
    */
    ASSERT_EQ(output, output_expected.flatten());
}

TEST(fconv, fconv_dilated) {
    const size_t len = 128;
    float data[len];
    for (size_t i = 0; i < len; i++) {
        data[i] = i;
    }
    Mat im(8, 8, 2, data, DataType::Float);

    const size_t w_len = 3 * 3 * 2 * 2;
    float weight_data[w_len];
    for (size_t i = 0; i < w_len; i++) {
        weight_data[i] = i;
    }
    Mat weight(w_len, weight_data, DataType::Float);

    // TODO: Auto calculate output shape
    const size_t output_len = 32;
    Mat output(output_len, DataType::Float);
    Mat output_expected(4, 4, 2, DataType::Float);
    for (size_t i = 0; i < output_len; i++) {
        output_expected[i] = 0;
    }

    fconv(im, weight, 3, 3, 2, 2, 2, 2, 2, 2, 2, output);

    baseline_fconv(im, weight, 3, 3, 2, 2, 2, 2, 2, 2, 2, output_expected);

    ASSERT_EQ(output, output_expected.flatten());
}

}  // namespace bnn

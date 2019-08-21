// Copyright 2019 JD.com Inc. JD AI

#include <gtest/gtest.h>

#include <common/helper.h>
#include <dabnn/net.h>

/*
TEST(net, bireal18cifar) {
    float input[3 * 224 * 224];
    FORZ(i, 3*224*224) {
        input[i] = 1;
    }

    const std::string blob_name = "187";
    {
    auto net1 = bnn::Net::create();
    net1->read("/data/local/tmp/model_cifar100.dab");
    net1->optimize = true;
    net1->run(input);
    const auto &blob1 = net1->get_blob(blob_name);
    ASSERT_NEAR((*blob1)[0], -2.3525, 1e-4);
    ASSERT_NEAR((*blob1)[1], -2.0001, 1e-4);
    ASSERT_NEAR((*blob1)[2], -18.6939, 1e-4);
    }
}
*/

TEST(net, bireal18imagenet_comparison) {
    float input[3 * 224 * 224];
    FORZ(i, 3 * 224 * 224) { input[i] = 1; }

    const std::string blob_name = "188";
    std::shared_ptr<bnn::Mat> blob1, blob2;
    {
        auto net = bnn::Net::create();
        net->read("/data/local/tmp/model_imagenet.dab");
        net->optimize = false;
        net->run(input);
        blob1 = net->get_blob(blob_name);
    }
    {
        auto net = bnn::Net::create();
        net->read("/data/local/tmp/model_imagenet.dab");
        net->optimize = true;
        net->run(input);
        blob2 = net->get_blob(blob_name);
    }
    ASSERT_EQ(*blob1, *blob2);
}

TEST(net, bireal18imagenet) {
    float input[3 * 224 * 224];
    FORZ(i, 3 * 224 * 224) { input[i] = 1; }

    const std::string blob_name = "188";
    {
        auto net = bnn::Net::create();
        net->read("/data/local/tmp/model_imagenet.dab");
        net->optimize = true;
        net->run(input);
        const auto blob = net->get_blob(blob_name);
        ASSERT_NEAR((*blob)[0], -0.9431, 1e-4);
        ASSERT_NEAR((*blob)[1], -1.2626, 1e-4);
        ASSERT_NEAR((*blob)[2], -5.1064, 1e-4);
    }
}

TEST(net, bireal18imagenetstem_comparison) {
    float input[3 * 224 * 224];
    FORZ(i, 3 * 224 * 224) { input[i] = 1; }

    const std::string blob_name = "216";
    std::shared_ptr<bnn::Mat> blob1, blob2;
    {
        auto net = bnn::Net::create();
        net->read("/data/local/tmp/model_imagenet_stem.dab");
        net->optimize = false;
        net->run(input);
        blob1 = net->get_blob(blob_name);
    }
    {
        auto net = bnn::Net::create();
        net->read("/data/local/tmp/model_imagenet_stem.dab");
        net->optimize = true;
        net->run(input);
        blob2 = net->get_blob(blob_name);
    }
    ASSERT_EQ(*blob1, *blob2);
}

TEST(net, bireal18imagenetstem) {
    float input[3 * 224 * 224];
    FORZ(i, 3 * 224 * 224) { input[i] = 1; }

    const std::string blob_name = "216";
    {
        auto net = bnn::Net::create();
        net->read("/data/local/tmp/model_imagenet_stem.dab");
        net->optimize = true;
        net->run(input);
        const auto &blob = net->get_blob(blob_name);
        ASSERT_NEAR((*blob)[0], 1.9842, 1e-4);
        ASSERT_NEAR((*blob)[1], 3.4204, 1e-4);
        ASSERT_NEAR((*blob)[2], -3.2586, 1e-4);
    }
}

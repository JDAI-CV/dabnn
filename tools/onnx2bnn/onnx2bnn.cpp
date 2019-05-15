// Copyright 2019 JD.com Inc. JD AI

#include <fstream>
#include <map>
#include <numeric>
#include <string>

#include <common/StrKeyMap.h>
#include <glog/logging.h>
#include "NodeAttrHelper.h"
#include "OnnxConverter.h"
#include <common/argh.h>
#include "common/log_helper.h"

using std::string;
using std::vector;

void usage(const std::string &filename) {
    std::cout << "Usage: " << filename << " onnx_model output_filename" << std::endl;
}

int main(int argc, char **argv) {
    argh::parser cmdl(argc, argv);
    google::InitGoogleLogging(cmdl[0].c_str());
    FLAGS_alsologtostderr = true;
    if (!cmdl(2)) {
        usage(cmdl[0]);
        return -1;
    }
    bnn::OnnxConverter::Level opt_level = bnn::OnnxConverter::Level::kModerate;
    if (cmdl["strict"]) {
        opt_level = bnn::OnnxConverter::Level::kStrict;
    }
    if (cmdl["aggresive"]) {
        opt_level = bnn::OnnxConverter::Level::kAggressive;
    }

    ONNX_NAMESPACE::ModelProto model_proto;
    {
        std::ifstream ifs(cmdl[1], std::ios::in | std::ios::binary);
        model_proto.ParseFromIstream(&ifs);
        ifs.close();
    }

    bnn::OnnxConverter converter;
    converter.Convert(model_proto, cmdl[2], opt_level);

    google::protobuf::ShutdownProtobufLibrary();
    return 0;
}

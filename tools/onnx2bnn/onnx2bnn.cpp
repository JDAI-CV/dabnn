// Copyright 2019 JD.com Inc. JD AI

#include <fstream>
#include <map>
#include <numeric>
#include <string>

#include <common/StrKeyMap.h>
#include <common/argh.h>
#include <glog/logging.h>
#include "NodeAttrHelper.h"
#include "OnnxConverter.h"
#include "common/log_helper.h"

using std::string;
using std::vector;

void usage(const std::string &filename) {
    std::cout
        << "Usage: " << filename
        << " onnx_model output_filename [--optimize strict|moderate|aggressive]"
        << std::endl;
    std::cout << "Example: " << filename
              << " model.onnx model.dab (The optimization leval will be "
                 "\"aggressive\")"
              << std::endl;
    std::cout << "Example: " << filename
              << " model.onnx model.dab --optimize strict (The optimization "
                 "level will be \"strict\")"
              << std::endl;
}

int main(int argc, char **argv) {
    argh::parser cmdl;
    cmdl.add_param("optimize");
    cmdl.parse(argc, argv);
    google::InitGoogleLogging(cmdl[0].c_str());
    FLAGS_alsologtostderr = true;
    if (!cmdl(2)) {
        usage(cmdl[0]);
        return -1;
    }
    // flags like 'onnx2bnn --strict' is not supported now
    for (const auto flag : cmdl.flags()) {
        std::cout << "Invalid flag: " << flag << std::endl;
        usage(cmdl[0]);
        return -2;
    }

    const std::string opt_level_str =
        cmdl("optimize").str().empty() ? "aggressive" : cmdl("optimize").str();

    bnn::OnnxConverter::Level opt_level;
    if (opt_level_str == "strict") {
        opt_level = bnn::OnnxConverter::Level::kStrict;
    } else if (opt_level_str == "moderate") {
        opt_level = bnn::OnnxConverter::Level::kModerate;
    } else if (opt_level_str == "aggressive") {
        opt_level = bnn::OnnxConverter::Level::kAggressive;
    } else {
        std::cout << "Invalid optimization level: " << opt_level_str
                  << std::endl;
        usage(cmdl[0]);
        return -3;
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

// Copyright 2019 JD.com Inc. JD AI

#include <fstream>
#include <map>
#include <numeric>
#include <string>

#include <common/StrKeyMap.h>
#include <glog/logging.h>
#include "NodeAttrHelper.h"
#include "OnnxConverter.h"
#include "common/log_helper.h"

using std::string;
using std::vector;

void usage(const std::string &filename) {
    std::cout << "Usage: " << filename << " onnx_model output_filename" << std::endl;
}

int main(int argc, char **argv) {
    google::InitGoogleLogging(argv[0]);
    FLAGS_alsologtostderr = true;
    if (argc != 3) {
        usage(argv[0]);
        return -1;
    }
    ONNX_NAMESPACE::ModelProto model_proto;
    {
        std::ifstream ifs(argv[1], std::ios::in | std::ios::binary);
        model_proto.ParseFromIstream(&ifs);
        ifs.close();
    }

    bnn::OnnxConverter converter;
    converter.Convert(model_proto, argv[2]);

    google::protobuf::ShutdownProtobufLibrary();
    return 0;
}

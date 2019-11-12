// Copyright 2019 JD.com Inc. JD AI

#include <common/StrKeyMap.h>
#include <common/argh.h>
#include <glog/logging.h>

#include <fstream>
#include <map>
#include <numeric>
#include <string>

#include "NodeAttrHelper.h"
#include "OnnxConverter.h"
#include "common/log_helper.h"

using std::string;
using std::vector;

void usage(const std::string &filename) {
    std::cout << "Usage:" << std::endl;
    std::cout << "  " << filename
              << " onnx_model output_filename [ --strict | --moderate | "
                 "--aggressive ] [--binary-convs list] [--binary-convs-file "
                 "filename] [--exclude-first-last] [--verbose]"
              << std::endl;
    std::cout << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout
        << "  --aggressive          The default optimization level. In this "
           "level, "
           "onnx2bnn will mark all convolutions with binary (+1/-1) weights as "
           "binary convolutions. It is for the existing BNN models, which may "
           "not use the correct padding value. Note: The output of the "
           "generated dabnn model is different from that of the ONNX model "
           "since the padding value is 0 instead of -1."
        << std::endl;
    std::cout << "  --moderate            This level is for our \"standard\" "
                 "implementation -- A Conv operator with binary weight and "
                 "following a -1 Pad operator."
              << std::endl;
    std::cout
        << "  --strict              In this level, onnx2bnn only recognizes "
           "the "
           "following natural and correct \"pattern\" of binary convolutions: "
           "A Conv operator, whose input is got from a Sign op and a Pad op "
           "(the order doesn't matter), and weight is got from a Sign op."
        << std::endl;
    std::cout
        << "  --binary-convs-file   A text file containing the **output "
           "names** of some convolutions, which will be treated as binary "
           "convlutions unconditionally. It is mainly for benchmark purpose."
        << std::endl;
    std::cout
        << "  --binary-convs        A ','-sperated list (for example, "
           "\"4,5,10\") containing the **output "
           "names** of some convolutions, which will be treated as binary "
           "convlutions unconditionally. It is mainly for benchmark purpose."
        << std::endl;
    std::cout
        << "  --exclude-first-last  Set all convolutions except the first and "
           "last convolution as binary convoslutions regardless of what they "
           "actually are. It is mainly for benchmark purpose."
        << std::endl;
    std::cout << std::endl;
    std::cout << "Example:" << std::endl;
    std::cout << "  " << filename
              << " model.onnx model.dab (The optimization leval will be "
                 "\"aggressive\")"
              << std::endl;
    std::cout << "  " << filename
              << " model.onnx model.dab --strict (The optimization "
                 "level will be \"strict\")"
              << std::endl;
}

vector<string> split(string s, string delimiter) {
    vector<string> parts;
    size_t pos = 0;
    std::string token;
    while ((pos = s.find(delimiter)) != std::string::npos) {
        token = s.substr(0, pos);
        std::cout << token << std::endl;
        s.erase(0, pos + delimiter.length());
    }
    parts.push_back(s);
    return parts;
}

int main(int argc, char **argv) {
    argh::parser cmdl;
    cmdl.add_params({"--binary-convs", "--binary-convs-file"});
    cmdl.parse(argc, argv);
    google::InitGoogleLogging(cmdl[0].c_str());
    FLAGS_alsologtostderr = true;
    if (!cmdl(2)) {
        usage(cmdl[0]);
        return -1;
    }
    for (const auto flag : cmdl.flags()) {
        if (flag != "strict" && flag != "moderate" && flag != "aggressive" &&
            flag != "verbose" && flag != "exclude-first-last") {
            std::cout << "Invalid flag: " << flag << std::endl;
            usage(cmdl[0]);
            return -2;
        }
    }
    int manual_binary_list = 0;
    if (cmdl["binary-convs"]) {
        manual_binary_list++;
    }
    if (cmdl["binary-convs-file"]) {
        manual_binary_list++;
    }
    if (cmdl["exclude-first-last"]) {
        manual_binary_list++;
    }
    if (manual_binary_list > 1) {
        std::cerr << "--binary--convs, --binary-convs-list and "
                     "--exclude-first-last are mutually exclusive"
                  << std::endl;
        return -2;
    }

    bnn::OnnxConverter::Level opt_level =
        bnn::OnnxConverter::Level::kAggressive;
    if (cmdl["strict"]) {
        opt_level = bnn::OnnxConverter::Level::kStrict;
    } else if (cmdl["moderate"]) {
        opt_level = bnn::OnnxConverter::Level::kModerate;
    } else if (cmdl["aggressive"]) {
        opt_level = bnn::OnnxConverter::Level::kAggressive;
    }

    if (cmdl["verbose"]) {
        FLAGS_v = 5;
    }

    vector<string> expected_binary_conv_outputs;
    const auto binary_list_filepath = cmdl("binary-convs-file").str();
    if (!binary_list_filepath.empty()) {
        std::ifstream ifs(binary_list_filepath);
        if (ifs.is_open()) {
            string binary_conv_output;
            while (ifs >> binary_conv_output) {
                expected_binary_conv_outputs.push_back(binary_conv_output);
            }
        } else {
            std::cerr << "Cannot open file \"" + binary_list_filepath + "\""
                      << std::endl;
            return -1;
        }
    }
    const auto binary_convs_str = cmdl("binary-convs").str();
    if (!binary_convs_str.empty()) {
        expected_binary_conv_outputs = split(binary_convs_str, ",");
    }
    bool exclude_first_last = false;
    if (cmdl["exclude-first-last"]) {
        exclude_first_last = true;
    }

    ONNX_NAMESPACE::ModelProto model_proto;
    {
        const auto model_filepath = cmdl[1];
        std::ifstream ifs(model_filepath, std::ios::in | std::ios::binary);
        if (ifs.fail()) {
            std::cerr << "The file \"" + model_filepath + "\" doesn't exist"
                      << std::endl;
            return -1;
        } else if (!model_proto.ParseFromIstream(&ifs)) {
            std::cerr << "Failed to parse file \"" + model_filepath + "\""
                      << std::endl;
            return -2;
        }
        ifs.close();
    }
    if (exclude_first_last) {
        vector<ONNX_NAMESPACE::NodeProto> binary_node_candidates;
        for (const auto &node : model_proto.graph().node()) {
            if (node.op_type() == "Conv" || node.op_type() == "Gemm") {
                binary_node_candidates.push_back(node);
            }
        }
        for (size_t i = 0; i < binary_node_candidates.size(); i++) {
            if (i == 0 || i == binary_node_candidates.size() - 1) {
                continue;
            }
            expected_binary_conv_outputs.push_back(
                binary_node_candidates[i].output(0));
        }
    }

    bnn::OnnxConverter converter;
    const auto binary_conv_outputs = converter.Convert(
        model_proto, cmdl[2], opt_level, expected_binary_conv_outputs);

    LOG(INFO) << "Conversion completed! Found " << binary_conv_outputs.size()
              << " binary convolutions. Add --verbose to get what they are.";
    VLOG(5) << "The outputs name of binary convolutions are: ";
    for (const auto &x : binary_conv_outputs) {
        VLOG(5) << x;
    }

    google::protobuf::ShutdownProtobufLibrary();
    return 0;
}

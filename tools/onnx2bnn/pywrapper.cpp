#include <pybind11/pybind11.h>

#include "OnnxConverter.h"

namespace py = pybind11;

void convert(const std::string &model_str, const std::string &filepath,
             const std::string &level_str) {
    using namespace bnn;
    ONNX_NAMESPACE::ModelProto model_proto;
    bool ret = model_proto.ParseFromString(model_str);
    if (!ret) {
        throw std::invalid_argument("Read protobuf string failed");
    }

    OnnxConverter::Level level = OnnxConverter::Level::kModerate;

    if (level_str == "moderate") {
        level = OnnxConverter::Level::kModerate;
    } else if (level_str == "strict") {
        level = OnnxConverter::Level::kStrict;
    } else if (level_str == "aggressive") {
        level = OnnxConverter::Level::kAggressive;
    } else {
        throw std::invalid_argument(
            "Level can only be moderate, strict or aggressive");
    }
    OnnxConverter converter;
    converter.Convert(model_proto, filepath, level);
    google::protobuf::ShutdownProtobufLibrary();
}

PYBIND11_MODULE(_onnx2bnn, m) { m.def("convert", &convert, ""); }

// Copyright 2019 JD.com Inc. JD AI

#include "OnnxConverter.h"

#include <bitset>
#include <fstream>
#include <map>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include <common/Shaper.h>
#include <common/StrKeyMap.h>
#include <common/common_bitpack.h>
#include <common/flatbuffers_helper.h>
#include <common/helper.h>
#include <glog/logging.h>
#include <onnx/onnx_pb.h>
#include <onnx/optimizer/optimize.h>
#include "NodeAttrHelper.h"

namespace bnn {

using std::string;
using std::unique_ptr;
using std::vector;
using Shape = Shaper::Shape;

bool is_binary_weight(const float *data, Shape shape);

std::string OnnxConverter::m(const std::string &str) {
    if (name_map_.find(str) != name_map_.end()) {
        return name_map_[str];
    }

    return str;
}

void OnnxConverter::AddBinConv(const std::string &input_name,
                               const std::vector<int> &strides,
                               const std::vector<int> &pads,
                               const std::vector<int> &dilations, int group,
                               const std::string &weight_name,
                               const std::string &output_name,
                               BTensor bin_weight) {
    bnn_bin_tensors_[weight_name] = bin_weight;

    css bin_name = input_name + "_bin";

    {
        const auto param = flatbnn::CreateBinarizeDirect(
            builder_, input_name.c_str(), bin_name.c_str());
        const auto layer =
            flatbnn::CreateLayer(builder_, flatbnn::LayerType::Binarize, 0, 0,
                                 0, 0, 0, 0, 0, 0, 0, 0, param);
        layers_.push_back(layer);
    }

    if (group != 1) {
        BNN_ASSERT(group == 2, "");
        css shuffle_output = output_name + "_shuffle";
        vector<string> split_outputs;
        vector<string> group_conv_weights;
        vector<string> group_conv_outputs;
        FORZ(g, group) {
            split_outputs.push_back(output_name + "_split_" +
                                    std::to_string(g));
            group_conv_weights.push_back(weight_name + "_group_" +
                                         std::to_string(g));
            group_conv_outputs.push_back(output_name + "_group_" +
                                         std::to_string(g));
        }

        {
            const auto param = flatbnn::CreateShuffleDirect(
                builder_, bin_name.c_str(), shuffle_output.c_str());
            const auto layer =
                flatbnn::CreateLayer(builder_, flatbnn::LayerType::Shuffle, 0,
                                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, param);
            layers_.push_back(layer);
        }

        LOG(INFO) << "Shuffle added";

        {
            const auto fbs_outputs = pack_str_vec(split_outputs, builder_);

            const auto param = flatbnn::CreateSplitDirect(
                builder_, shuffle_output.c_str(), &fbs_outputs);
            const auto layer =
                flatbnn::CreateLayer(builder_, flatbnn::LayerType::Split, 0, 0,
                                     0, 0, 0, 0, 0, 0, 0, 0, 0, param);
            layers_.push_back(layer);
        }

        LOG(INFO) << "Split added";

        {
            const auto weights = split(bin_weight, group);
            FORZ(g, group) {
                const auto param = flatbnn::CreateBinConv2DDirect(
                    builder_, split_outputs[g].c_str(),
                    group_conv_weights[g].c_str(), nullptr, &pads, &strides,
                    &dilations, group_conv_outputs[g].c_str());
                const auto layer = flatbnn::CreateLayer(
                    builder_, flatbnn::LayerType::BinConv2D, 0, param);
                const auto flat_tensor = flatbnn::CreateTensorDirect(
                    builder_, flatbnn::DataType::Bit, &weights[g].data, nullptr,
                    &weights[g].shape, group_conv_weights[g].c_str());
                tensors_.push_back(flat_tensor);
                layers_.push_back(layer);
            }
        }

        LOG(INFO) << "Conv added";

        {
            const auto fbs_inputs = pack_str_vec(group_conv_outputs, builder_);
            const auto param = flatbnn::CreateConcatDirect(
                builder_, &fbs_inputs, 3, output_name.c_str());
            const auto layer =
                flatbnn::CreateLayer(builder_, flatbnn::LayerType::Concat, 0, 0,
                                     0, 0, 0, 0, 0, 0, param);
            layers_.push_back(layer);
        }

        LOG(INFO) << "Concat added";
    } else {
        const auto param = flatbnn::CreateBinConv2DDirect(
            builder_, bin_name.c_str(), weight_name.c_str(), nullptr, &pads,
            &strides, &dilations, output_name.c_str());
        const auto layer = flatbnn::CreateLayer(
            builder_, flatbnn::LayerType::BinConv2D, 0, param);
        const auto flat_tensor = flatbnn::CreateTensorDirect(
            builder_, flatbnn::DataType::Bit, &bin_weight.data, nullptr,
            &bin_weight.shape, weight_name.c_str());
        tensors_.push_back(flat_tensor);
        layers_.push_back(layer);
    }
}

void OnnxConverter::AddFloatConv(
    const string &input_name, const std::vector<int> &strides,
    const std::vector<int> &pads, const std::vector<int> &dilations, int group,
    const string &weight_name, const std::optional<std::string> &bias_name,
    const string &output_name, FTensor float_weight) {
    flatbuffers::Offset<flatbnn::Layer> layer;
    flatbuffers::Offset<flatbnn::Tensor> flat_tensor;

    if (group != 1) {
        // TODO: Support it
        throw std::invalid_argument("group != 1 is not supported");
    }

    LOG(INFO) << "Vanilla conv" + weight_name;
    bnn_tensors_[weight_name] = float_weight;

    auto param = flatbnn::CreateFpConv2DDirect(
        builder_, input_name.c_str(), weight_name.c_str(),
        bias_name ? bias_name.value().c_str() : nullptr, &pads, &strides,
        &dilations, output_name.c_str());
    layer = flatbnn::CreateLayer(builder_, flatbnn::LayerType::FpConv2D, param);
    flat_tensor = flatbnn::CreateTensorDirect(
        builder_, flatbnn::DataType::Float32, nullptr, &float_weight.data,
        &float_weight.shape, weight_name.c_str());
    tensors_.push_back(flat_tensor);
    layers_.push_back(layer);
}

void OnnxConverter::AddConv(const string &input_name,
                            const std::vector<int> &strides,
                            const std::vector<int> &pads,
                            const std::vector<int> &dilations, int group,
                            const string &ori_weight_name,
                            const std::optional<std::string> &bias_name,
                            const string &output_name) {
    flatbuffers::Offset<flatbnn::Layer> layer;

    flatbuffers::Offset<flatbnn::Tensor> flat_tensor;
    const auto &onnx_weight = onnx_float_tensors_.at(ori_weight_name);

    FTensor bnn_float_tensor = OnnxToBnn(onnx_weight);
    string weight_name = ori_weight_name + "_conv_w";
    shaper_.AddShape(weight_name, bnn_float_tensor.shape);
    shaper_.Conv(input_name, strides[1], strides[0], 1, 1, pads[2], pads[3],
                 pads[0], pads[1], weight_name, output_name);
    if (!is_binary_weight(onnx_weight.data.data(), onnx_weight.shape)) {
        AddFloatConv(input_name, strides, pads, dilations, group, weight_name,
                     bias_name, output_name, bnn_float_tensor);
    } else {
        // binary conv
        LOG(INFO) << "Binary conv" + weight_name;
        BTensor weight_tensor = bitpack(bnn_float_tensor);
        AddBinConv(input_name, strides, pads, dilations, group, weight_name,
                   output_name, weight_tensor);
    }
}

bool is_binary_weight(const float *data, Shape shape) {
    FORZ(i, Shaper::total(shape)) {
        if (data[i] != -1 && data[i] != 1) {
            return false;
        }
    }
    return true;
}

/*
 * Bitpack a bnn tensor, input_channels should be the last dimension
 */
OnnxConverter::BTensor OnnxConverter::bitpack(OnnxConverter::FTensor ftensor) {
    static_assert(std::is_same<bin_t, uint64_t>::value,
                  "bitpack requires bin_t is 64 bit");

    auto c = Shaper::kc(ftensor.shape);

    BNN_ASSERT(c % 64 == 0, ftensor.shape);

    vector<bin_t> packed_data;
    // if (c % 128 == 0) {
    if (false) {
        const auto size = Shaper::total(ftensor.shape);
        packed_data.resize(size / 64);
        pack_128_fallback(&ftensor.data[0], &packed_data[0], size);
    } else {
        bin_t tmp;

        FORZS(i, Shaper::total(ftensor.shape), 64) {
            pack_64_bitset(&ftensor.data[i], &tmp);
            packed_data.push_back(tmp);
        }
    }

    Shape shape = {ftensor.shape[0], ftensor.shape[1], ftensor.shape[2],
                   ftensor.shape[3]};
    return {packed_data, shape};
}

std::vector<OnnxConverter::BTensor> OnnxConverter::split(
    OnnxConverter::BTensor input, int num) {
    std::vector<BTensor> outputs;
    const auto shape = input.get_shape_for_accessing_element();
    BNN_ASSERT(Shaper::kn(shape) % num == 0, "");
    const auto n_per_group = Shaper::kn(shape) / num;
    FORZ(i, num) {
        BTensor tensor;
        FORZ(n, n_per_group) {
            FORZ(h, Shaper::kh(shape)) {
                FORZ(w, Shaper::kw(shape)) {
                    FORZ(c, Shaper::kc(shape)) {
                        const auto &tmp =
                            input.get({i * n_per_group + n, h, w, c});
                        tensor.data.push_back(tmp);
                    }
                }
            }
        }
        tensor.shape = input.shape;
        tensor.shape[0] = n_per_group;
        outputs.push_back(tensor);
    }
    return outputs;
}

vector<bin_t> bitpack(const float *data, Shape shape) {
    static_assert(std::is_same<bin_t, uint64_t>::value,
                  "bitpack requires bin_t is 64 bit");

    auto c = Shaper::onnx_kc(shape);

    BNN_ASSERT(c % 64 == 0, shape);

    vector<bin_t> packed;

    bin_t tmp;

    FORZS(i, Shaper::total(shape), 64) {
        pack_64_bitset(&data[i], &tmp);
        packed.push_back(tmp);
    }
    BNN_ASSERT(false, "");

    return packed;
}

void OnnxConverter::Convert(const ONNX_NAMESPACE::ModelProto &model_proto,
                            const std::string &filepath) {
    GOOGLE_PROTOBUF_VERIFY_VERSION;

    model_proto_ = ONNX_NAMESPACE::optimization::Optimize(
        model_proto, vector<string>{"eliminate_nop_pad"});

    for (const auto &tensor : model_proto_.graph().initializer()) {
        if (tensor.data_type() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
            Shape shape;
            for (auto dim : tensor.dims()) {
                shape.push_back(static_cast<uint32_t>(dim));
            }
            const float *ptr =
                tensor.float_data().empty()
                    ? reinterpret_cast<const float *>(tensor.raw_data().data())
                    : tensor.float_data().data();
            auto data_vec = vector<float>(ptr, ptr + Product(shape));

            onnx_float_tensors_[tensor.name()] = {data_vec, shape};
        }
        operands_.push_back(tensor.name());
    }
    LOG(INFO) << "We get " << onnx_bin_tensors_.size() << " binary weight and "
              << onnx_float_tensors_.size() << " float weight";

    vector<flatbuffers::Offset<flatbnn::Input>> inputs;
    for (const auto &input : model_proto_.graph().input()) {
        if (std::find(operands_.begin(), operands_.end(), input.name()) !=
            operands_.end()) {
            continue;
        }

        Shape shape;
        for (const auto &dim : input.type().tensor_type().shape().dim()) {
            if (dim.value_case() ==
                ONNX_NAMESPACE::TensorShapeProto_Dimension::kDimValue) {
                shape.push_back(static_cast<uint32_t>(dim.dim_value()));
            } else {
                throw std::invalid_argument(
                    "The input of graph doesn't have dim_value");
            }
        }
        Shape nnapi_shape{shape[0], shape[2], shape[3], shape[1]};
        shaper_.AddShape(input.name(), nnapi_shape);
        auto flat_input = flatbnn::CreateInputDirect(builder_, &nnapi_shape,
                                                     input.name().c_str());
        inputs.push_back(flat_input);
    }

    vector<string> skipped_act;
    bool has_reshape = false;
    for (const auto &node : model_proto_.graph().node()) {
        if (has_reshape) {
            throw std::invalid_argument(
                "Reshape can only be the last layer for now");
        }
        NodeAttrHelper helper(node);
        const auto &op = node.op_type();
        LOG(INFO) << "Node " << node.name();
        if (op == "Conv") {
            LOG(INFO) << "Start converting Conv";
            auto strides = helper.get("strides", vector<int>{1, 1});
            auto pads = helper.get("pads", vector<int>{0, 0, 0, 0});
            auto dilations = helper.get("dilations", vector<int>{1, 1});
            CHECK_EQ(pads.size(), 4ul);
            CHECK_EQ(strides.size(), 2ul);
            CHECK_EQ(dilations.size(), 2ul);
            auto group = helper.get("group", 1);
            std::optional<string> bias_name;
            if (node.input_size() >= 3) {
                auto ori_bias_name = m(node.input(2));
                bias_name = ori_bias_name + "_conv_b";
                bnn_tensors_[bias_name.value()] =
                    onnx_float_tensors_.at(ori_bias_name);
                auto flat_tensor = flatbnn::CreateTensorDirect(
                    builder_, flatbnn::DataType::Float32, nullptr,
                    &bnn_tensors_.at(bias_name.value()).data,
                    &bnn_tensors_.at(bias_name.value()).shape,
                    bias_name.value().c_str());
                tensors_.push_back(flat_tensor);
            }

            auto ori_weight_name = m(node.input(1));
            AddConv(m(node.input(0)), strides, pads, dilations, group,
                    ori_weight_name, bias_name, m(node.output(0)));
            LOG(INFO) << "Converting Conv completed";
        } else if (op == "AveragePool" || op == "MaxPool" ||
                   op == "GlobalAveragePool" || op == "GlobalMaxPool") {
            LOG(INFO) << "Start converting Pool";
            auto input_name = m(node.input(0));
            auto output_name = m(node.output(0));
            vector<int> strides, pads, kernel_shape;
            if (op == "AveragePool" || op == "MaxPool") {
                strides = helper.get("strides", vector<int>{1, 1});
                pads = helper.get("pads", vector<int>{0, 0, 0, 0});
                kernel_shape = helper.get("kernel_shape", vector<int>{0, 0});
                auto count_include_pad = helper.get("count_include_pad", 0);
                if (count_include_pad == 1) {
                    throw std::invalid_argument(
                        "count_include_pad == 1 is not supported");
                }
                auto storage_order = helper.get("storage_order", 0);
                if (storage_order == 1) {
                    throw std::invalid_argument(
                        "storage_order == 1 is not supported");
                }
                if (helper.has_attr("auto_pad")) {
                    throw std::invalid_argument("auto_pad is not supported");
                }
            } else {
                strides = {0, 0};
                pads = {0, 0, 0, 0};
                kernel_shape = {-1, -1};  // -1 for global
            }
            CHECK_EQ(pads.size(), 4ul);
            CHECK_EQ(kernel_shape.size(), 2ul);
            CHECK_EQ(strides.size(), 2ul);
            shaper_.Pool(input_name, strides[1], strides[0], pads[2], pads[3],
                         pads[0], pads[1], kernel_shape[0], kernel_shape[1],
                         output_name);
            flatbuffers::Offset<flatbnn::Layer> layer;
            if (op == "AveragePool" || op == "GlobalAveragePool") {
                auto param = flatbnn::CreateAvePoolDirect(
                    builder_, input_name.c_str(), &kernel_shape, &pads,
                    &strides, output_name.c_str());
                layer = flatbnn::CreateLayer(
                    builder_, flatbnn::LayerType::AvePool, 0, 0, param);
            } else {
                auto param = flatbnn::CreateMaxPoolDirect(
                    builder_, input_name.c_str(), &kernel_shape, &pads,
                    &strides, output_name.c_str());
                layer = flatbnn::CreateLayer(
                    builder_, flatbnn::LayerType::MaxPool, 0, 0, 0, param);
            }
            layers_.push_back(layer);
            LOG(INFO) << "Converting Pool completed";
        } else if (op == "Relu") {
            LOG(INFO) << "Start converting Relu";
            auto input_name = m(node.input(0));
            auto output_name = m(node.output(0));
            shaper_.Relu(input_name, output_name);
            auto param = flatbnn::CreateReluDirect(builder_, input_name.c_str(),
                                                   output_name.c_str());
            auto layer = flatbnn::CreateLayer(
                builder_, flatbnn::LayerType::Relu, 0, 0, 0, 0, param);
            layers_.push_back(layer);
            LOG(INFO) << "Converting Relu completed";
        } else if (op == "Add") {
            LOG(INFO) << "Start converting Add";
            auto input1_name = m(node.input(0));
            auto input2_name = m(node.input(1));
            auto output_name = m(node.output(0));
            shaper_.Eltwise(input1_name, input2_name, output_name);
            auto param = flatbnn::CreateAddDirect(builder_, input1_name.c_str(),
                                                  input2_name.c_str(),
                                                  output_name.c_str());
            auto layer = flatbnn::CreateLayer(builder_, flatbnn::LayerType::Add,
                                              0, 0, 0, 0, 0, 0, 0, param);
            layers_.push_back(layer);
            LOG(INFO) << "Converting Add completed";
        } else if (op == "Gemm") {
            LOG(INFO) << "Start converting Gemm";
            auto transA = helper.get("transA", 0);
            auto transB = helper.get("transB", 0);
            auto alpha = helper.get("alpha", 1.0f);
            auto beta = helper.get("beta", 1.0f);
            if (transA == 0 && transB == 1 && alpha == 1.f && beta == 1.f) {
                auto input_name = m(node.input(0));
                auto weight_name = m(node.input(1));
                {
                    bnn_tensors_[weight_name] =
                        onnx_float_tensors_.at(weight_name);
                    const auto &weight_tensor = bnn_tensors_[weight_name];
                    shaper_.AddShape(weight_name, weight_tensor.shape);
                    auto flat_tensor = flatbnn::CreateTensorDirect(
                        builder_, flatbnn::DataType::Float32, nullptr,
                        &weight_tensor.data, &weight_tensor.shape,
                        weight_name.c_str());
                    tensors_.push_back(flat_tensor);
                }
                string bias_name;
                if (node.input_size() >= 3) {
                    bias_name = m(node.input(2));
                    bnn_tensors_[bias_name] = onnx_float_tensors_.at(bias_name);
                    const auto &bias_tensor = bnn_tensors_[bias_name];
                    auto flat_tensor = flatbnn::CreateTensorDirect(
                        builder_, flatbnn::DataType::Float32, nullptr,
                        &bias_tensor.data, &bias_tensor.shape,
                        bias_name.c_str());
                    tensors_.push_back(flat_tensor);
                }
                auto output_name = m(node.output(0));
                shaper_.FC(input_name, weight_name, output_name);
                auto param = flatbnn::CreateFCDirect(
                    builder_, input_name.c_str(), weight_name.c_str(),
                    node.input_size() >= 3 ? bias_name.c_str() : nullptr,
                    output_name.c_str());
                auto layer =
                    flatbnn::CreateLayer(builder_, flatbnn::LayerType::FC, 0, 0,
                                         0, 0, 0, 0, param, 0);
                layers_.push_back(layer);
            } else {
                throw std::invalid_argument(
                    "Only transA == 0, transB == 1, alpha == 1.0 and beta == "
                    "1.0 is supported.");
            }
            LOG(INFO) << "Converting Gemm completed";
        } else if (op == "Softmax") {
            LOG(INFO) << "Start converting Softmax";
            auto input_name = m(node.input(0));
            auto output_name = m(node.output(0));
            shaper_.Softmax(input_name, output_name);
            // simply ignore attribute "axis", because nnapi softmax didn't has
            // this attr, and we will check the equality of the two ops in
            // DaqReader.cpp
            auto param = flatbnn::CreateSoftmaxDirect(
                builder_, input_name.c_str(), output_name.c_str());
            auto layer = flatbnn::CreateLayer(
                builder_, flatbnn::LayerType::Softmax, 0, 0, 0, 0, 0, param);
            layers_.push_back(layer);
            LOG(INFO) << "Converting Softmax completed";
        } else if (op == "Concat") {
            LOG(INFO) << "Start converting Concat";
            vector<std::string> concat_inputs_str;
            for (const auto &onnx_input : node.input()) {
                concat_inputs_str.push_back(m(onnx_input));
            }
            vector<flatbuffers::Offset<flatbuffers::String>> concat_inputs =
                pack_str_vec(concat_inputs_str, builder_);
            auto axis = helper.get("axis", 1);
            uint32_t axis_nchw_to_nhwc[4]{0, 3, 1, 2};
            auto output_name = m(node.output(0));
            shaper_.Concat(concat_inputs_str, axis, output_name);
            auto param = flatbnn::CreateConcatDirect(builder_, &concat_inputs,
                                                     axis_nchw_to_nhwc[axis],
                                                     output_name.c_str());
            auto layer =
                flatbnn::CreateLayer(builder_, flatbnn::LayerType::Concat, 0, 0,
                                     0, 0, 0, 0, 0, 0, param);
            layers_.push_back(layer);
            LOG(INFO) << "Converting Concat completed";
        } else if (op == "Dropout") {
            LOG(INFO) << "Start converting Dropout";
            // Dropout does nothing, so the output is the same as the input
            name_map_[node.output(0)] = m(node.input(0));
            LOG(INFO) << "Converting Dropout completed";
        } else if (op == "Reshape") {
            LOG(INFO) << "Start converting Reshape";
            has_reshape = true;
            LOG(INFO) << "Converting Reshape completed";
        } else if (op == "BatchNormalization") {
            LOG(INFO) << "Start converting BatchNormalization";
            const auto &input_name = node.input(0);
            const auto &output_name = node.output(0);

            const auto coeff_a_name = output_name + "_a";
            const auto coeff_b_name = output_name + "_b";

            CalculateCoeff(node, coeff_a_name, coeff_b_name);

            shaper_.Affine(input_name, output_name);
            auto param = flatbnn::CreateAffineDirect(
                builder_, input_name.c_str(), coeff_a_name.c_str(),
                coeff_b_name.c_str(), output_name.c_str());
            auto layer =
                flatbnn::CreateLayer(builder_, flatbnn::LayerType::Affine, 0, 0,
                                     0, 0, 0, 0, 0, 0, 0, param);
            layers_.push_back(layer);

            auto a_tensor = flatbnn::CreateTensorDirect(
                builder_, flatbnn::DataType::Float32, nullptr,
                &onnx_float_tensors_[coeff_a_name].data,
                &onnx_float_tensors_[coeff_a_name].shape, coeff_a_name.c_str());
            auto b_tensor = flatbnn::CreateTensorDirect(
                builder_, flatbnn::DataType::Float32, nullptr,
                &onnx_float_tensors_[coeff_b_name].data,
                &onnx_float_tensors_[coeff_b_name].shape, coeff_b_name.c_str());
            tensors_.push_back(a_tensor);
            tensors_.push_back(b_tensor);
            LOG(INFO) << "Converting BatchNormalization completed";
        } else {
            throw std::invalid_argument("Unsupported operator " + op);
        }
    }
    auto flat_layers = builder_.CreateVector(layers_);
    auto flat_inputs = builder_.CreateVector(inputs);
    auto flat_tensors = builder_.CreateVector(tensors_);
    auto flat_model =
        flatbnn::CreateModel(builder_, flat_layers, flat_tensors, flat_inputs);

    builder_.Finish(flat_model);

    LOG(INFO) << "Shapes: ";
    LOG(INFO) << shaper_;

    std::ofstream ofs(filepath);
    ofs.write(reinterpret_cast<char *>(builder_.GetBufferPointer()),
              builder_.GetSize());
    ofs.close();
}

void OnnxConverter::CalculateCoeff(const ONNX_NAMESPACE::NodeProto &node,
                                   const std::string &coeff_a_name,
                                   const std::string &coeff_b_name) {
    const auto &scale_name = node.input(1);
    const auto &b_name = node.input(2);
    const auto &mean_name = node.input(3);
    const auto &var_name = node.input(4);
    const auto &eps = NodeAttrHelper(node).get("eps", 1e-5f);

    const auto &scale = onnx_float_tensors_.at(scale_name);
    const auto &b = onnx_float_tensors_.at(b_name);
    const auto &mean = onnx_float_tensors_.at(mean_name);
    const auto &var = onnx_float_tensors_.at(var_name);

    std::vector<float> coeff_a_data, coeff_b_data;
    FORZ(i, scale.data.size()) {
        const float tmp = std::sqrt(var.data[i] + eps);
        coeff_a_data.push_back(scale.data[i] / tmp);
        coeff_b_data.push_back(b.data[i] - scale.data[i] * mean.data[i] / tmp);
    }

    for (const auto &node2 : model_proto_.graph().node()) {
        if (node2.op_type() == "Conv" && node2.output(0) == node.input(0)) {
            const auto &weight = onnx_float_tensors_[node2.input(1)];
            if (is_binary_weight(weight.data.data(), weight.shape)) {
                LOG(INFO) << "I found!";
                {
                    int channels = Shaper::onnx_kc(weight.shape);
                    int width = Shaper::onnx_kw(weight.shape);
                    int height = Shaper::onnx_kh(weight.shape);

                    FORZ(i, coeff_b_data.size()) {
                        coeff_b_data[i] = coeff_b_data[i] + channels * width *
                                                                height *
                                                                coeff_a_data[i];
                    }
                }
                {
                    FORZ(i, coeff_a_data.size()) { coeff_a_data[i] *= -2; }
                }
            }
        }
    }

    FTensor coeff_a;
    coeff_a.data = coeff_a_data;
    coeff_a.shape = Shape{static_cast<Shaper::len_t>(coeff_a_data.size())};
    FTensor coeff_b;
    coeff_b.data = coeff_b_data;
    coeff_b.shape = Shape{static_cast<Shaper::len_t>(coeff_b_data.size())};
    onnx_float_tensors_[coeff_a_name] = coeff_a;
    onnx_float_tensors_[coeff_b_name] = coeff_b;
}

}  // namespace bnn

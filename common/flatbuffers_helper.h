// Copyright 2019 JD.com Inc. JD AI

//
// Created by daquexian on 8/22/18.
//

#ifndef DNNLIBRARY_FLATBUFFERS_HELPER_H
#define DNNLIBRARY_FLATBUFFERS_HELPER_H

#include <string>
#include <vector>

#include <common/dab_generated.h>
#include <flatbuffers/flatbuffers.h>

#include <common/helper.h>

#define UNPACK(name) const auto name = unpack_fbs(param->name());

#define UNPACK_LAYER(name, ...)                \
    const auto *param = layer->name##_param(); \
    FOR_EACH(UNPACK, __VA_ARGS__)

#define ADD_LAYER_MULTI_OUTPUTS(name, shape_func, ...)                      \
    const auto *param = layer->name##_param();                              \
    BNN_ASSERT(param != nullptr, "");                                       \
    FOR_EACH(UNPACK, __VA_ARGS__)                                           \
    shaper.shape_func(__VA_ARGS__);                                         \
    for (const auto output : LAST_ARG(__VA_ARGS__)) {                       \
        if (mat_map_.find(output) == mat_map_.end()) {                      \
            const auto &output_shape = shaper[output];                      \
            const auto &input_mat =                                         \
                *mat_map_[get_input(FIRST_ARG(__VA_ARGS__))];               \
            add_mat(output,                                                 \
                    std::make_shared<Mat>(output_shape[1], output_shape[2], \
                                          output_shape[3],                  \
                                          input_mat.data_type, output));    \
        }                                                                   \
    }

#define ADD_LAYER(name, shape_func, ...)                                      \
    const auto *param = layer->name##_param();                                \
    BNN_ASSERT(param != nullptr, "");                                         \
    FOR_EACH(UNPACK, __VA_ARGS__)                                             \
    if (mat_map_.find(LAST_ARG(__VA_ARGS__)) == mat_map_.end()) {             \
        shaper.shape_func(__VA_ARGS__);                                       \
        const auto &output_shape = shaper[LAST_ARG(__VA_ARGS__)];             \
        const auto &input_mat = *mat_map_[get_input(FIRST_ARG(__VA_ARGS__))]; \
        add_mat(LAST_ARG(__VA_ARGS__),                                        \
                std::make_shared<Mat>(output_shape[1], output_shape[2],       \
                                      output_shape[3], input_mat.data_type,   \
                                      LAST_ARG(__VA_ARGS__)));                \
    }

#define ADD_LAYER_WITH_DATA_TYPE(name, shape_func, mat_data_type, ...)  \
    const auto *param = layer->name##_param();                          \
    BNN_ASSERT(param != nullptr, "");                                   \
    FOR_EACH(UNPACK, __VA_ARGS__)                                       \
    if (mat_map_.find(LAST_ARG(__VA_ARGS__)) == mat_map_.end()) {       \
        shaper.shape_func(__VA_ARGS__);                                 \
        const auto &output_shape = shaper[LAST_ARG(__VA_ARGS__)];       \
        add_mat(LAST_ARG(__VA_ARGS__),                                  \
                std::make_shared<Mat>(output_shape[1], output_shape[2], \
                                      output_shape[3], mat_data_type,   \
                                      LAST_ARG(__VA_ARGS__)));          \
    }

#define ADD_INPLACE_LAYER(name, shape_func, ...)                      \
    const auto *param = layer->name##_param();                        \
    BNN_ASSERT(param != nullptr, "");                                 \
    FOR_EACH(UNPACK, __VA_ARGS__)                                     \
    add_mat(LAST_ARG(__VA_ARGS__), mat_map_[FIRST_ARG(__VA_ARGS__)]); \
    shaper.shape_func(__VA_ARGS__);

// quick fix
inline const std::string get_input(const std::vector<std::string> inputs) {
    return inputs[0];
}

inline const std::string get_input(const std::string input) { return input; }

inline uint32_t unpack_fbs(const uint32_t fbs) { return fbs; }

inline int32_t unpack_fbs(const int32_t fbs) { return fbs; }

inline std::string unpack_fbs(const flatbuffers::String *fbs) {
    if (fbs == nullptr) {
        return "";
    }
    return fbs->str();
}

inline std::vector<std::string> unpack_fbs(
    const flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>>
        *fbs_vec) {
    using fbsoff_t = flatbuffers::uoffset_t;
    std::vector<std::string> std_vec;
    for (size_t i = 0; i < fbs_vec->size(); i++) {
        std_vec.push_back(fbs_vec->Get(static_cast<fbsoff_t>(i))->str());
    }
    return std_vec;
}

inline std::vector<int32_t> unpack_fbs(
    const flatbuffers::Vector<int32_t> *fbs_vec) {
    using fbsoff_t = flatbuffers::uoffset_t;
    std::vector<int32_t> std_vec;
    for (size_t i = 0; i < fbs_vec->size(); i++) {
        std_vec.push_back(fbs_vec->Get(static_cast<fbsoff_t>(i)));
    }
    return std_vec;
}

inline std::vector<flatbuffers::Offset<flatbuffers::String>> pack_str_vec(
    const std::vector<std::string> &str_vec,
    flatbuffers::FlatBufferBuilder &_fbb) {
    std::vector<flatbuffers::Offset<flatbuffers::String>> fbs_str_vec;
    for (const auto &str : str_vec) {
        auto flat_input = _fbb.CreateString(str.c_str(), str.size());
        fbs_str_vec.push_back(flat_input);
    }
    return fbs_str_vec;
}

inline std::string layer_type_to_str(flatbnn::LayerType type) {
    switch (type) {
        case flatbnn::LayerType::FC:
            return "fc";
        case flatbnn::LayerType::Add:
            return "Add";
        case flatbnn::LayerType::Relu:
            return "relu";
        case flatbnn::LayerType::FpConv2D:
            return "fpconv";
        case flatbnn::LayerType::BinConv2D:
            return "binconv";
        case flatbnn::LayerType::Concat:
            return "concat";
        case flatbnn::LayerType::MaxPool:
            return "maxpool";
        case flatbnn::LayerType::AvePool:
            return "avepool";
        case flatbnn::LayerType::Softmax:
            return "softmax";
        case flatbnn::LayerType::Affine:
            return "affine";
        case flatbnn::LayerType::Binarize:
            return "binarize";
        case flatbnn::LayerType::Split:
            return "split";
        case flatbnn::LayerType::Shuffle:
            return "shuffle";
        default:
            BNN_ASSERT(false, "Missing type in this function");
    }
}

#endif  // DNNLIBRARY_FLATBUFFERS_HELPER_H

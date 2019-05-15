// Copyright 2019 JD.com Inc. JD AI

#ifndef BNN_ONNXCONVERTER_H
#define BNN_ONNXCONVERTER_H

#include <set>
#include "optional.h"

#include <common/Shaper.h>
#include <common/StrKeyMap.h>
#include <common/daq_generated.h>
#include <common/helper.h>
#include <glog/logging.h>
#include <onnx/onnx_pb.h>

namespace bnn {
class OnnxConverter {
   private:
    Shaper shaper_;

    template <typename T>
    struct Tensor {
        std::vector<T> data;
        Shaper::Shape shape;
        inline T get(const std::vector<Shaper::len_t> &x) {
            auto step = get_shape_for_accessing_element();
            for (int i = shape.size() - 2; i >= 0; i--) {
                step[i] *= step[i + 1];
            }
            step.push_back(1);
            step.erase(step.begin());
            Shaper::len_t idx = 0;
            FORZ(i, x.size()) { idx += x[i] * step[i]; }
            // PNT(x, size, get_shape_for_accessing_element(), idx,
            // Shaper::total(size));
            BNN_ASSERT(idx < Shaper::total(get_shape_for_accessing_element()),
                       "");
            return data[idx];
        }
        Shaper::Shape get_shape_for_accessing_element();
    };

    using FTensor = Tensor<float>;
    using BTensor = Tensor<bin_t>;

    ONNX_NAMESPACE::ModelProto model_proto_;

    std::map<std::string, std::string> name_map_;

    std::string m(const std::string &str);

    flatbuffers::FlatBufferBuilder builder_;

    std::vector<std::string> operands_;
    StrKeyMap<FTensor> bnn_tensors_;
    StrKeyMap<FTensor> onnx_float_tensors_;
    std::vector<flatbuffers::Offset<flatbnn::Layer>> layers_;

    std::vector<flatbuffers::Offset<flatbnn::Tensor>> tensors_;

    BTensor bitpack(FTensor ftensor);

    std::vector<BTensor> split(BTensor input, int num);

    void AddBinConv(const std::string &input_name,
                    const std::vector<int> &strides,
                    const std::vector<int> &pads,
                    const std::vector<int> &dilations, int group,
                    const std::string &weight_name,
                    const std::string &output_name, BTensor bin_weight);

    void AddFloatConv(const std::string &input_name,
                      const std::vector<int> &strides,
                      const std::vector<int> &pads,
                      const std::vector<int> &dilations, int group,
                      const std::string &weight_name,
                      const nonstd::optional<std::string> &bias_name,
                      const std::string &output_name, FTensor float_weight);

    void AddConv(const std::string &input_name, const std::vector<int> &strides,
                 const std::vector<int> &pads,
                 const std::vector<int> &dilations, int group,
                 const std::string &ori_weight_name,
                 const nonstd::optional<std::string> &bias_name,
                 const std::string &output_name,
                 const bool binary);

    void CalculateCoeff(const ONNX_NAMESPACE::NodeProto &node,
                        const std::string &coeff_a_name,
                        const std::string &coeff_b_name);

    void GetBinTensors();

    /**
     * onnx: [filter_out_channel, filter_in_channel / group, height, width]
     * nnapi: [1, height, width, depth_out]
     */
    template <typename T>
    Tensor<T> OnnxToNnapiDw(const Tensor<T> &src) {
        Tensor<T> dest;
        dest.data.resize(Product(src.shape));
        // t for total
        auto out_t = src.shape[0], in_t = src.shape[1], h_t = src.shape[2],
             w_t = src.shape[3];
        CHECK_EQ(in_t, 1u);
        for (uint32_t out = 0; out < out_t; out++) {
            for (uint32_t in = 0; in < in_t; in++) {
                for (uint32_t h = 0; h < h_t; h++) {
                    for (uint32_t w = 0; w < w_t; w++) {
                        auto onnx_idx = out * in_t * h_t * w_t +
                                        in * h_t * w_t + h * w_t + w;
                        auto nnapi_idx = h * w_t * out_t + w * out_t + out;
                        dest.data[nnapi_idx] = src.data[onnx_idx];
                    }
                }
            }
        }
        dest.shape = {in_t, h_t, w_t, out_t};
        return dest;
    }

    /**
     * onnx: [filter_out_channel, filter_in_channel, height, width]
     * bnn: [depth_out, height, width, depth_in]
     */
    template <typename T>
    Tensor<T> OnnxToBnn(const Tensor<T> &src) {
        Tensor<T> dest;
        dest.data.resize(Product(src.shape));
        // t for total
        auto out_t = src.shape[0], in_t = src.shape[1], h_t = src.shape[2],
             w_t = src.shape[3];
        for (uint32_t out = 0; out < out_t; out++) {
            for (uint32_t in = 0; in < in_t; in++) {
                for (uint32_t h = 0; h < h_t; h++) {
                    for (uint32_t w = 0; w < w_t; w++) {
                        auto onnx_idx = out * in_t * h_t * w_t +
                                        in * h_t * w_t + h * w_t + w;
                        auto nnapi_idx = out * h_t * w_t * in_t +
                                         h * w_t * in_t + w * in_t + in;
                        dest.data[nnapi_idx] = src.data[onnx_idx];
                    }
                }
            }
        }
        dest.shape = {out_t, h_t, w_t, in_t};
        return dest;
    }

   public:
    enum class Level {
        kStrict,
        kSoft,
        kExtremeSoft,
    };
    void Convert(const ONNX_NAMESPACE::ModelProto &model,
                 const std::string &filepath,
                 const Level level=Level::kSoft);
};

template <>
inline Shaper::Shape
OnnxConverter::Tensor<float>::get_shape_for_accessing_element() {
    return shape;
}

template <>
inline Shaper::Shape
OnnxConverter::Tensor<bin_t>::get_shape_for_accessing_element() {
    BNN_ASSERT(shape.size() == 4, "");
    auto ret = shape;
    ret[3] /= 64;
    return ret;
}
}  // namespace bnn

#endif /* BNN_ONNXCONVERTER_H */

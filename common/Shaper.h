// Copyright 2019 JD.com Inc. JD AI

#ifndef DNN_SHAPER_H
#define DNN_SHAPER_H

#include <string>
#include <vector>

#include <iostream>

#include "StrKeyMap.h"
#include "helper.h"

namespace bnn {
class Shaper {
   public:
    using len_t = uint32_t;
    using Shape = std::vector<len_t>;
    static len_t onnx_kn(const Shape &shape) { return shape[0]; }
    static len_t onnx_kh(const Shape &shape) { return shape[2]; }
    static len_t onnx_kw(const Shape &shape) { return shape[3]; }
    static len_t onnx_kc(const Shape &shape) { return shape[1]; }
    static len_t kn(const Shape &shape) { return shape[0]; }
    static len_t kh(const Shape &shape) { return shape[1]; }
    static len_t kw(const Shape &shape) { return shape[2]; }
    static len_t kc(const Shape &shape) { return shape[3]; }
    static len_t n(const Shape &shape) { return shape[0]; }
    static len_t h(const Shape &shape) { return shape[1]; }
    static len_t w(const Shape &shape) { return shape[2]; }
    static len_t c(const Shape &shape) { return shape[3]; }
    static len_t total(const Shape &shape) { return Product(shape); }

    void Conv(const std::string &input_name, const std::vector<int32_t> strides,
              const std::vector<int32_t> dilations,
              const std::vector<int32_t> paddings,
              const std::string &weight_name, const std::string &output_name);
    void Conv(const std::string &input_name, const std::vector<int32_t> strides,
              const std::vector<int32_t> dilations,
              const std::vector<int32_t> paddings,
              const std::string &weight_name, const std::string &bias_name,
              const std::string &output_name);
    void Conv(const std::string &input_name, int32_t strideX, int32_t strideY,
              int32_t dilationX, int32_t dilationY, int32_t paddingLeft,
              int32_t paddingRight, int32_t paddingTop, int32_t paddingBottom,
              const std::string &weight_name, const std::string &output_name);
    void DepthwiseConv(const std::string &input_name,
                       const std::vector<int32_t> strides,
                       const std::vector<int32_t> dilations,
                       const std::vector<int32_t> paddings,
                       const std::string &weight_name,
                       const std::string &output_name);
    void DepthwiseConv(const std::string &input_name, int32_t strideX,
                       int32_t strideY, int32_t dilationX, int32_t dilationY,
                       int32_t paddingLeft, int32_t paddingRight,
                       int32_t paddingTop, int32_t paddingBottom,
                       const std::string &weight_name,
                       const std::string &output_name);
    void StridedSlice(const std::string &input_name,
                      const std::vector<int32_t> &starts,
                      const std::vector<int32_t> &ends,
                      const std::vector<int32_t> &strides, int32_t beginMask,
                      int32_t endMask, int32_t shrinkAxisMask,
                      const std::string &output_name);
    void Pool(const std::string &input_name, const std::vector<int32_t> strides,
              const std::vector<int32_t> paddings,
              const std::vector<int32_t> kernel_shape,
              const std::string &output_name);
    void Pool(const std::string &input_name, int32_t strideX, int32_t strideY,
              int32_t paddingLeft, int32_t paddingRight, int32_t paddingTop,
              int32_t paddingBottom, int32_t height, int32_t width,
              const std::string &output_name);
    void Softmax(const std::string &input_name, const std::string &output_name);
    void Relu(const std::string &input_name, const std::string &output_name);
    void Concat(const std::vector<std::string> &input_names, uint32_t axis,
                const std::string &output_name);
    void LRN(const std::string &input_name, const std::string &output_name);
    void FC(const std::string &input_name, const std::string &weight_name,
            const std::string &output_name);
    void Eltwise(const std::string &input1_name, const std::string &input2_name,
                 const std::string &output_name);
    void Eltwise(const std::string &input1_name,
                 const std::string &output_name);
    void Affine(const std::string &input_name, const std::string &output_name);
    void Affine(const std::string &input_name, const std::string &a,
                const std::string &b, const std::string &output_name);
    void BatchToSpace(const std::string &input_name,
                      const std::vector<int32_t> &block_sizes,
                      const std::string &output_name);
    void SpaceToBatch(const std::string &input_name,
                      const std::vector<int32_t> &block_sizes,
                      const std::vector<int32_t> &pads,
                      const std::string &output_name);
    void Binarize(css &input_name, css &output_name);
    void Split(const std::string &input_name,
               const std::vector<std::string> &output_names, uint32_t axis = 3);
    void Shuffle(css &input_name, css &output_name);
    void AddShape(const std::string &name, const Shape &shape);
    size_t GetSize(const std::string &name);
    void Clear();

    inline const Shape &operator[](const std::string &key) const {
        return shape_map_.at(key);
    }
    inline friend std::ostream &operator<<(std::ostream &os,
                                           const Shaper &shaper) {
        for (const auto &p : shaper.shape_map_) {
            os << (p.first + ": ") << p.second << std::endl;
        }
        return os;
    }

   private:
    StrKeyMap<Shape> shape_map_;
};
}  // namespace bnn

#endif

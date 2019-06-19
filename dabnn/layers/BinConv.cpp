// Copyright 2019 JD.com Inc. JD AI

#include "BinConv.h"

#include <common/baseline.h>
#include <dabnn/bconv.h>
#include <dabnn/bgemm.h>
#include <dabnn/bitpack.h>
#include <dabnn/fused_binarize_im2col.h>
#include <dabnn/net.h>
#include <dabnn/pad.h>

namespace bnn {

BinConv::BinConv(NetCP net, const std::string &name, css input, css weight,
                 css output, int pad_h, int pad_w, int stride_h, int stride_w)
    : Layer(net, name, "Bin Conv"),
      input_mat(mat(input)),
      weight_mat(mat(weight)),
      output_mat(mat(output)),
      pad_h(pad_h),
      pad_w(pad_w),
      stride_h(stride_h),
      stride_w(stride_w) {
    auto &mat_map = net.lock()->mat_map_;
    const auto binaized_name = "binaized_for_" + output + "_cal";
    if (mat_map.find(binaized_name) == mat_map.end()) {
        auto &input_mat = *mat_map[input];
        mat_map[binaized_name] = std::make_shared<Mat>(
            input_mat.h, input_mat.w, input_mat.elem_c,
            DataType::Bit, binaized_name);
    }
    binarized_mat = mat(binaized_name);

    const auto pad_name = "pad_for_" + output + "_cal";
    if (mat_map.find(pad_name) == mat_map.end()) {
        auto &input_mat = *mat_map[input];
        mat_map[pad_name] = std::make_shared<Mat>(
            input_mat.h + pad_h * 2, input_mat.w + pad_w * 2, input_mat.elem_c,
            DataType::Bit, pad_name);
    }
    padded_mat = mat(pad_name);

    const auto col_mat_name = "col_mat";
    if (mat_map.find(col_mat_name) == mat_map.end()) {
        const auto len = output_mat->h * output_mat->w * weight_mat->h *
                         weight_mat->w * input_mat->elem_c;
        mat_map[col_mat_name] = std::make_shared<Mat>(len, bnn::DataType::Bit);
    }
    col_mat = mat(col_mat_name);

    if (net.lock()->optimize && !direct_conv_compatible() &&
        gemm_compatible()) {
        const auto trans_weight_mat_name = "trans_" + weight;
        // transpose the weight for bgemm
        const int m = weight_mat->n;
        BNN_ASSERT(weight_mat->total() % m == 0, "");
        const int k = weight_mat->total() / m;
        transposed_weight_mat =
            std::make_shared<Mat>(m, k * 64, DataType::Bit);
        auto *trans_data_ptr =
            static_cast<uint64_t *>(transposed_weight_mat->data);
        auto *data_ptr = static_cast<uint64_t *>(weight_mat->data);
        FORZ(i, k) {
            FORZ(j, m) { 
                BNN_ASSERT(static_cast<size_t>(i * m + j) < transposed_weight_mat->total(), i * m + j, " ", transposed_weight_mat->total());
                trans_data_ptr[i * m + j] = data_ptr[j * k + i]; 
            }
        }
        net_.lock()->add_mat(trans_weight_mat_name, transposed_weight_mat);
    }
}

bool BinConv::direct_conv_compatible() const {
#ifdef __aarch64__
    if (weight_mat->h == 3 && weight_mat->w == 3 && input_mat->c == 1 &&
        stride_h == stride_w) {
        return true;
    }
    if (weight_mat->h == 3 && weight_mat->w == 3 && input_mat->c == 2 &&
        stride_h == stride_w) {
        return true;
    }
    if (weight_mat->h == 3 && weight_mat->w == 3 && input_mat->c == 4 &&
        stride_h == stride_w) {
        return true;
    }
    if (weight_mat->h == 3 && weight_mat->w == 3 && input_mat->c == 8 &&
        stride_h == stride_w) {
        return true;
    }
    if (weight_mat->h == 3 && weight_mat->w == 3 && input_mat->c == 16 &&
        stride_h == stride_w) {
        return true;
    }
    return false;
#else
    return false;
#endif
}

bool BinConv::gemm_compatible() const {
#ifdef __ARM_NEON
    return weight_mat->h * weight_mat->w * weight_mat->c % 2 == 0;
#else
    return false;
#endif
}

void BinConv::forward_impl() const {
    if (net_.lock()->optimize) {
        if (direct_conv_compatible()) {
            pack_mat(*input_mat, *binarized_mat);
            pad(*binarized_mat, pad_h, pad_w, *padded_mat);
            bconv_3x3(*padded_mat, *weight_mat, *output_mat, stride_h);
        } else if (gemm_compatible()) {
            output_mat->fill<float>(0.f);
            bnn::fused_binarize_im2col(*input_mat, weight_mat->h, weight_mat->w, pad_h, pad_w, stride_h, stride_w, 1, 1, *col_mat);
            const int m = weight_mat->n;
            const int n = output_mat->h * output_mat->w;
            const int k = weight_mat->h * weight_mat->w * weight_mat->c;
            bgemm(m, n, k, static_cast<uint64_t *>(transposed_weight_mat->data),
                  m, static_cast<uint64_t *>(col_mat->data), k,
                  static_cast<float *>(output_mat->data), m);
        } else {
            pack_mat(*input_mat, *binarized_mat);
            baseline_bconv(*binarized_mat, *weight_mat, weight_mat->h,
                           weight_mat->w, pad_h, pad_w, stride_h, stride_w, 1,
                           1, output_mat->c, *output_mat);
        }
    } else {
        pack_mat(*input_mat, *binarized_mat);
        baseline_bconv(*binarized_mat, *weight_mat, weight_mat->h, weight_mat->w,
                       pad_h, pad_w, stride_h, stride_w, 1, 1, output_mat->c,
                       *output_mat);
    }
}

std::string BinConv::to_str() const {
    std::stringstream ss;
    ss << type_ << ", ";
    PNT_TO(ss, input_mat->h, input_mat->w, input_mat->elem_c, weight_mat->h,
           weight_mat->w, weight_mat->n, pad_h, pad_w);

    return ss.str();
}

}  // namespace bnn

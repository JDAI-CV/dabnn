// Copyright 2019 JD.com Inc. JD AI

#ifndef BNN_NET_H
#define BNN_NET_H

#include <map>
#include <memory>

#include <common/Shaper.h>
#include <common/daq_generated.h>
#include <common/helper.h>
#include <dabnn/layers/Add.h>
#include <dabnn/layers/Affine.h>
#include <dabnn/layers/AvePool.h>
#include <dabnn/layers/BinConv.h>
#include <dabnn/layers/FloatConv.h>
#include <dabnn/layers/MaxPool.h>
#include "layer.h"
#include "mat.h"

namespace bnn {
class Net : public std::enable_shared_from_this<Net> {
   private:
#ifdef BNN_BENCHMARK
    std::map<std::string, double> layer_time_;
#endif
    StrKeyMap<std::shared_ptr<Mat>> mat_map_;
    Shaper shaper;
    void add_mat(const std::string &name, std::shared_ptr<Mat> mat);
    std::vector<std::shared_ptr<std::vector<float>>> float_bufs;
    std::vector<std::shared_ptr<Layer>> layers;

    std::string input_name_;

    std::weak_ptr<Net> get_weak();

    void read_impl(const void *ptr);

    Net() = default;

    friend class Layer;
    friend class BinConv;
    friend class AvePool;
    friend class MaxPool;
    friend class FloatConv;
    friend class Affine;
    friend class Add;

   public:
    void read(const std::string &path);
    void read_buf(const void *ptr);
    void prepare();
    void run(void *input);
    static std::shared_ptr<Net> create();
    const flatbnn::Model *model_;

    std::shared_ptr<Mat> get_blob(const std::string &name);
    bool optimize = true;
    bool run_fconv = true;
    bool strict = true;

#ifdef BNN_BENCHMARK
    void print_time();
#endif
};
}  // namespace bnn

#endif /* BNN_NET_H */

// Copyright 2019 JD.com Inc. JD AI

#ifndef BNN_LAYER_H
#define BNN_LAYER_H

#include <memory>
#include <string>
#include <vector>
#include "mat.h"

namespace bnn {

class Net;

class Layer {
   protected:
    // handy alias
    using NetCP = const std::weak_ptr<Net>;
    using MatCP = const std::shared_ptr<Mat>;
    using MatP = std::shared_ptr<Mat>;
    MatCP mat(const std::string &name) const;

   public:
    Layer(NetCP net, const std::string &name, const std::string &type)
        : net_(net), name_(name), type_(type) {}
    // virtual destructor
    virtual ~Layer();

    NetCP net_;

    void forward();
    virtual void forward_impl() const = 0;
    virtual std::string to_str() const;

    // layer name
    std::string name_;
    // layer type name
    std::string type_;
};

}  // namespace bnn

#endif  // BNN_LAYER_H

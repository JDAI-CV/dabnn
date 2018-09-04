// Copyright 2019 JD.com Inc. JD AI

#ifndef BNN_NET_FOR_JNI_H
#define BNN_NET_FOR_JNI_H

#include <android/asset_manager_jni.h>

#include <dabnn/net.h>

namespace bnn {
class NetJNI {
   public:
    NetJNI() : net_(Net::create()){};
    const std::shared_ptr<Net> net_;
    AAsset* asset;
};
}  // namespace bnn

#endif /* BNN_NET_FOR_JNI_H */

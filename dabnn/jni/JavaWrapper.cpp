// Copyright 2019 JD.com Inc. JD AI

#include "jni_handle.h"

#include <map>
#include <vector>

#include <android/asset_manager_jni.h>

#include <dabnn/jni/net_for_jni.h>

using std::map;
using std::string;

jint throwException(JNIEnv *env, std::string message);

extern "C" JNIEXPORT void JNICALL
Java_me_daquexian_dabnn_Net_initHandle(JNIEnv *env, jobject obj /* this */) {
    bnn::NetJNI *net = new bnn::NetJNI();
    setHandle(env, obj, net);
}

extern "C" JNIEXPORT jobject JNICALL Java_me_daquexian_dabnn_Net_readAsset(
    JNIEnv *env, jobject obj /* this */, jobject javaAssetManager,
    jstring javaFilename) {
    using Net = bnn::NetJNI;
    Net *net = getHandle<Net>(env, obj);

    string filename = string(env->GetStringUTFChars(javaFilename, nullptr));
    AAssetManager *mgrr = AAssetManager_fromJava(env, javaAssetManager);

    AAsset *asset =
        AAssetManager_open(mgrr, filename.c_str(), AASSET_MODE_BUFFER);
    net->asset = asset;
    const uint8_t *buf = static_cast<const uint8_t *>(AAsset_getBuffer(asset));
    BNN_ASSERT(buf != nullptr, "");
    net->net_->read_buf(buf);
    return obj;
}

extern "C" JNIEXPORT jobject JNICALL Java_me_daquexian_dabnn_Net_readFile(
    JNIEnv *env, jobject obj /* this */, jstring javaFilename) {
    using Net = bnn::NetJNI;
    Net *net = getHandle<Net>(env, obj);

    string filename = string(env->GetStringUTFChars(javaFilename, nullptr));
    net->net_->read(filename);
    return obj;
}

extern "C" JNIEXPORT jfloatArray JNICALL Java_me_daquexian_dabnn_Net_getBlob(
    JNIEnv *env, jobject obj /* this */, jstring javaBlobName) {
    using Net = bnn::NetJNI;
    Net *net = getHandle<Net>(env, obj);
    string blobName = string(env->GetStringUTFChars(javaBlobName, nullptr));
    const auto blob = net->net_->get_blob(blobName);
    jsize output_len = blob->total();
    jfloatArray result = env->NewFloatArray(output_len);
    env->SetFloatArrayRegion(result, 0, output_len,
                             static_cast<jfloat *>(*blob));

    return result;
}

extern "C" JNIEXPORT jobject JNICALL Java_me_daquexian_dabnn_Net_predict(
    JNIEnv *env, jobject obj /* this */, jfloatArray dataArrayObject) {
    using Net = bnn::NetJNI;
    Net *net = getHandle<Net>(env, obj);

    jfloat *data = env->GetFloatArrayElements(dataArrayObject, nullptr);

    net->net_->run(data);

    return obj;
}

extern "C" JNIEXPORT void JNICALL
Java_me_daquexian_dabnn_Net_dispose(JNIEnv *env, jobject obj /* this */) {
    using Net = bnn::NetJNI;
    auto handle = getHandle<Net>(env, obj);
    if (handle != nullptr) {
        if (handle->asset != nullptr) {
            AAsset_close(handle->asset);
            handle->asset = nullptr;
        }
        delete handle;
        setHandle(env, obj, nullptr);
    }
}

jint throwException(JNIEnv *env, std::string message) {
    jclass exClass;
    std::string className = "java/lang/RuntimeException";

    exClass = env->FindClass(className.c_str());

    return env->ThrowNew(exClass, message.c_str());
}

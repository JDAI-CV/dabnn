#! /usr/bin/env bash

set -e

MY_ANDROID_HOME="${ANDROID_HOME:-$HOME/Android/Sdk}"
MY_ANDROID_NDK_HOME="${ANDROID_NDK_HOME:-$MY_ANDROID_HOME/ndk-bundle}"
JNI_BUILD_DIR=build_jni_tmp
rm -rf ${JNI_BUILD_DIR} && mkdir ${JNI_BUILD_DIR} && pushd ${JNI_BUILD_DIR}
cmake -DCMAKE_SYSTEM_NAME=Android -DCMAKE_TOOLCHAIN_FILE=${MY_ANDROID_NDK_HOME}/build/cmake/android.toolchain.cmake -DANDROID_CPP_FEATURES=exceptions -DANDROID_PLATFORM=android-21 -DANDROID_ABI=arm64-v8a -DBNN_BUILD_JNI=ON -DBNN_BUILD_TEST=OFF -DBNN_BUILD_BENCHMARK=OFF ..
cmake --build . -- "-j$(nproc)"
popd
mkdir -p ci/android_aar/dabnn/src/main/jniLibs/arm64-v8a
cp ${JNI_BUILD_DIR}/dabnn/jni/libdabnn-jni.so ci/android_aar/dabnn/src/main/jniLibs/arm64-v8a/

# Increase version code and update version name

if (($# == 0)); then
    ret=0; tag=`git describe --exact-match --tags` || ret=$?
    if [ $ret != 0 ]; then
        echo "HEAD is not tagged, skip deploy aar"
        exit 0
    fi
    # tag is expected to be something like "v0.2", so remove the leading "v"
    ver=`echo $tag | cut -c 2-10`
elif (( $# == 1 )); then
    ver=$1
fi
echo "ver=$ver"

sed -i -E "s/versionName .+/versionName \"v$ver\"/" ci/android_aar/dabnn/build.gradle
sed -i -E "s/publishVersion = .+/publishVersion = \'$ver\'/" ci/android_aar/dabnn/build.gradle

pushd ci/android_aar
ANDROID_HOME=$MY_ANDROID_HOME ./gradlew clean build

# Publishing is only for myself
if [[ -z $BINTRAY_KEY ]]; then
    echo "BINTRAY_KEY is not set, skip bintray upload"
else
	echo "Publishing.."
	ANDROID_HOME=$MY_ANDROID_HOME ./gradlew bintrayUpload -PbintrayUser=daquexian566 -PbintrayKey=$BINTRAY_KEY
fi
popd
